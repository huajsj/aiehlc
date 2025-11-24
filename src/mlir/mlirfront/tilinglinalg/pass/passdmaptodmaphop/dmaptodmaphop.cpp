/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "dmaptodmaphop.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include <sstream>
#include <iostream>

using namespace mlir;
using namespace dmap;
using namespace dmaphop;

namespace {

// This pattern converts a dmap::FuncOp into a standard func::FuncOp and
// triggers the conversion of the operations inside.
struct DmapFuncOpLowering : public OpConversionPattern<dmap::FuncOp> {
    using OpConversionPattern<dmap::FuncOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(dmap::FuncOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto funcType = rewriter.getFunctionType({}, {});
        auto funcOp = rewriter.create<func::FuncOp>(op.getLoc(), op.getSymName(), funcType);

        rewriter.inlineRegionBefore(op.getBody(), funcOp.getBody(), funcOp.end());

        // Convert the dmap::YieldOp terminator to a func::ReturnOp
        for (auto &block : funcOp.getBlocks()) {
            if (auto yieldOp = dyn_cast_or_null<dmap::YieldOp>(block.getTerminator())) {
                rewriter.setInsertionPoint(yieldOp);
                rewriter.replaceOpWithNewOp<func::ReturnOp>(yieldOp);
            }
        }
        
        rewriter.eraseOp(op);
        return success();
    }
};


// Enum to distinguish between push and pull dataflow directions.
enum class DataflowDirection { Push, Pull };

// Generic function to lower a data movement operation (push or pull).
static LogicalResult lowerDataMovementOp(Operation *op, ConversionPatternRewriter &rewriter,
                                         RoutingTopology &router, DataflowDirection direction) {
    auto loc = op->getLoc();
    
    // Get topology info - core tile start row
    int core_start_row = (int) router.getRM()->getrsc()->absTileRow(TileType::Core, 0);
    
    // dmap.push %data, %stream -> stream is operand 1
    // dmap.pull %data from %stream -> stream is operand 1
    Value streamValue = op->getOperand(1);
    Operation *streamOp = streamValue.getDefiningOp();

    dmap::define_io_engine shimEngine, memEngine;
    dmap::define_core_group sourceCoreGroup, destCoreGroup;
    
    // Helper lambda to extract io_engine or core_group from a config operation
    auto extractConfig = [](Value configValue, 
                           dmap::define_io_engine &ioEngine, 
                           dmap::define_core_group &coreGroup) -> bool {
        if (auto ioConfig = configValue.getDefiningOp<dmap::create_io_engin_with_config>()) {
            ioEngine = ioConfig.getIoengine().getDefiningOp<dmap::define_io_engine>();
            return true;  // is IO engine
        } else if (auto cgConfig = configValue.getDefiningOp<dmap::create_core_group_with_config>()) {
            coreGroup = cgConfig.getCoregroup().getDefiningOp<dmap::define_core_group>();
            return false;  // is core group
        }
        return false;
    };

    // --- 1. Extract tile/engine definitions from the stream ---
    if (auto chainStreamOp = dyn_cast<dmap::createchainstream>(streamOp)) {
        if (chainStreamOp.getStreams().size() != 2) return op->emitError("only supports 2-hop chained streams");
        auto stream1 = chainStreamOp.getStreams()[0].getDefiningOp<dmap::createstream>();
        auto stream2 = chainStreamOp.getStreams()[1].getDefiningOp<dmap::createstream>();

        // Extract configurations dynamically
        dmap::define_io_engine io1, io2, io3;
        dmap::define_core_group cg1, cg2;
        
        bool s1_src_is_io = extractConfig(stream1.getSource(), io1, cg1);
        bool s1_dst_is_io = extractConfig(stream1.getDestination(), io2, cg2);
        bool s2_src_is_io = extractConfig(stream2.getSource(), shimEngine, sourceCoreGroup);
        bool s2_dst_is_io = extractConfig(stream2.getDestination(), memEngine, destCoreGroup);
        
        // Determine which is shim, mem, and core group based on the configuration
        // For chain streams, typically: SHIM -> MEM -> CORES or CORES -> MEM -> SHIM
        if (s1_src_is_io && s1_dst_is_io) {
            // stream1: IO -> IO (shim -> mem or mem -> shim)
            shimEngine = io1;
            memEngine = io2;
        }
        // stream2 configurations already extracted into shimEngine/memEngine/sourceCoreGroup/destCoreGroup

    } else if (auto createStreamOp = dyn_cast<dmap::createstream>(streamOp)) {
        // Extract source and destination dynamically
        dmap::define_io_engine srcIO, dstIO;
        dmap::define_core_group srcCG, dstCG;
        
        bool src_is_io = extractConfig(createStreamOp.getSource(), srcIO, srcCG);
        bool dst_is_io = extractConfig(createStreamOp.getDestination(), dstIO, dstCG);
        
        if (src_is_io && dst_is_io) {
            // IO -> IO: first is shim, second could be mem
            shimEngine = srcIO;
            memEngine = dstIO;
        } else if (src_is_io && !dst_is_io) {
            // IO -> CoreGroup (Push direction)
            shimEngine = srcIO;
            destCoreGroup = dstCG;
        } else if (!src_is_io && dst_is_io) {
            // CoreGroup -> IO (Pull direction)
            sourceCoreGroup = srcCG;
            shimEngine = dstIO;
        } else {
            // CoreGroup -> CoreGroup
            sourceCoreGroup = srcCG;
            destCoreGroup = dstCG;
        }
    } else {
        return op->emitError("unsupported stream type for lowering");
    }
    
    // Determine the actual core group to use based on direction
    dmap::define_core_group coreGroup = destCoreGroup ? destCoreGroup : sourceCoreGroup;
    
    // Validate we have the required components
    if (!shimEngine) {
        return op->emitError("no IO engine (shim) found in stream configuration");
    }
    if (!coreGroup) {
        return op->emitError("no core group found in stream configuration");
    }


    // --- 2. Build core tile list for createDataIO ---
    SmallVector<Point, 4> coreTilePoints;
    SmallVector<dmaphop::tile, 4> coreTiles;
    SmallVector<Value, 4> corePortsInValues;
    SmallVector<dmaphop::port, 4> corePortsOutOps;
    
    for (int i = 0; i < coreGroup.getCoreCount(); ++i) {
        // Use core_start_row as the base for core tile row calculation
        int row = (coreGroup.getGroupAxis() == "col") ? (i + core_start_row) : (coreGroup.getGroupIdx() + core_start_row);
        int col = (coreGroup.getGroupAxis() == "row") ? i : coreGroup.getGroupIdx();
        
        coreTilePoints.push_back(Point{row, col});
        
        auto coreTile = rewriter.create<dmaphop::tile>(loc, rewriter.getStringAttr("core"), rewriter.getI64IntegerAttr(col), rewriter.getI64IntegerAttr(row));
        coreTiles.push_back(coreTile);

        std::string inPortName = "corePortIn" + std::to_string(i);
        auto portInOp = rewriter.create<dmaphop::port>(loc, coreTile, rewriter.getStringAttr("In"), rewriter.getStringAttr(inPortName), rewriter.getI64IntegerAttr(0));
        corePortsInValues.push_back(portInOp.getResult());

        std::string outPortName = "corePortOut" + std::to_string(i);
        corePortsOutOps.push_back(rewriter.create<dmaphop::port>(loc, coreTile, rewriter.getStringAttr("Out"), rewriter.getStringAttr(outPortName), rewriter.getI64IntegerAttr(0)));
    }
    
    // --- 3. Use router to allocate shim column and channel ---
    // FIXED RULES:
    // Rule #1: Shim tile In and Out ports use the SAME channel number
    // Rule #2: Shim location (column) is determined by the destination core tile location
    //          - For Push (MM2S): shim -> cores, so destination is the FIRST core tile
    //          - For Pull (S2MM): cores -> shim, so shim destination relies on LAST core tile
    // Rule #3: Mem tile (if exists) is located in the SAME column as the shim tile
    
    static int ioIdx = 0;
    std::ostringstream dioNameStream;
    dioNameStream << "dio" << ioIdx++;
    
    // Get the target core tile for shim allocation
    // - For Push: shim is source, so use first core tile (destination)
    // - For Pull: shim is destination, so use last core tile (source)
    Point targetCoreTile = (direction == DataflowDirection::Push) 
                           ? coreTilePoints[0]                      // first tile for push
                           : coreTilePoints[coreTilePoints.size() - 1];  // last tile for pull
    std::optional<TypeBasedTileLoc> dstcoreloc(TypeBasedTileLoc{TileType::Core, targetCoreTile});
    
    // Determine DMA direction based on dataflow direction
    DMADIRECTION dmaDirection = (direction == DataflowDirection::Push) ? DMADIRECTION::MM2S : DMADIRECTION::S2MM;
    
    // Create DataIO - this allocates the optimal shim column and channel
    auto dio = router.createDataIO(dioNameStream.str(), dstcoreloc, dmaDirection);
    int shimcol = dio->colpos();
    int dioid = dio->id();
    int channel = dio->channel();
    
    std::cout << "Allocated DataIO: shim col=" << shimcol 
              << " channel=" << channel 
              << " IOID=" << dioid << std::endl;
    
    // --- 4. Create shim tile with allocated column ---
    // Note: Both shimPortOut and shimPortIn use the SAME channel (Rule #1)
    auto shimTile = rewriter.create<dmaphop::tile>(loc, rewriter.getStringAttr("shim"), 
                                                    rewriter.getI64IntegerAttr(shimcol), 
                                                    rewriter.getI64IntegerAttr(0));
    auto shimPortOut = rewriter.create<dmaphop::port>(loc, shimTile, rewriter.getStringAttr("Out"), 
                                                       rewriter.getStringAttr("shimPortOut"), 
                                                       rewriter.getI64IntegerAttr(channel));  // Same channel
    auto shimPortIn = rewriter.create<dmaphop::port>(loc, shimTile, rewriter.getStringAttr("In"), 
                                                      rewriter.getStringAttr("shimPortIn"), 
                                                      rewriter.getI64IntegerAttr(channel));   // Same channel
    
    // --- 5. Create mem tile if needed ---
    // Note: Mem tile is in the SAME column as shim tile (Rule #3)
    dmaphop::tile memTile;
    dmaphop::port memPortIn, memPortOut;
    if (memEngine) {
        // Mem tile uses the same column as shim (shimcol), not memEngine.getIoId()
        memTile = rewriter.create<dmaphop::tile>(loc, rewriter.getStringAttr("mem"), 
                                                  rewriter.getI64IntegerAttr(shimcol),  // Same col as shim (Rule #3)
                                                  rewriter.getI64IntegerAttr(0));
        memPortIn = rewriter.create<dmaphop::port>(loc, memTile, rewriter.getStringAttr("In"), 
                                                    rewriter.getStringAttr("memPortIn"), 
                                                    rewriter.getI64IntegerAttr(0));
        memPortOut = rewriter.create<dmaphop::port>(loc, memTile, rewriter.getStringAttr("Out"), 
                                                     rewriter.getStringAttr("memPortOut"), 
                                                     rewriter.getI64IntegerAttr(0));
    }
    

    // --- 6. Create dmaphop::create_hop Ops based on direction ---
    // Producer/Consumer semantics:
    // - Producers: ports that SOURCE data (send it out)
    //   * Core/Mem tiles: Out ports
    //   * Shim tile: In port (receives from external memory to send into fabric)
    // - Consumers: ports that SINK data (receive it)
    //   * Core/Mem tiles: In ports
    //   * Shim tile: Out port (sends to external memory from fabric)
    
    SmallVector<Value, 4> hops;
    SmallVector<Attribute, 4> producerPortSymbols;
    SmallVector<Attribute, 4> consumerPortSymbols;
    
    if (direction == DataflowDirection::Push) {
        // Push: SHIM (In) -> [MEM (Out)] -> CORES (In)
        // Producers: shimPortIn (for shim), memPortOut (if mem exists)
        // Consumers: all corePortIn
        
        if (memTile) { // SHIM -> MEM -> CORES
            hops.push_back(rewriter.create<dmaphop::create_hop>(loc, shimPortOut, memPortIn).getResult());
            hops.push_back(rewriter.create<dmaphop::create_hop>(loc, memPortOut, corePortsInValues[0]).getResult());
            // Producers: shimPortIn (shim receives from DDR), memPortOut (mem sends to cores)
            producerPortSymbols.push_back(SymbolRefAttr::get(rewriter.getContext(), "shimPortIn"));
            producerPortSymbols.push_back(SymbolRefAttr::get(rewriter.getContext(), "memPortOut"));
        } else { // SHIM -> CORES
            hops.push_back(rewriter.create<dmaphop::create_hop>(loc, shimPortOut, corePortsInValues[0]).getResult());
            // Producer: shimPortIn (shim receives from DDR to send into fabric)
            producerPortSymbols.push_back(SymbolRefAttr::get(rewriter.getContext(), "shimPortIn"));
        }
        
        // All core input ports are consumers
        for (size_t i = 0; i < corePortsInValues.size(); ++i) {
            std::string inPortName = "corePortIn" + std::to_string(i);
            consumerPortSymbols.push_back(SymbolRefAttr::get(rewriter.getContext(), inPortName));
        }
        
        // Create hops between cores (if multiple cores)
        for (size_t i = 0; i < corePortsOutOps.size() - 1; ++i) {
            hops.push_back(rewriter.create<dmaphop::create_hop>(loc, corePortsOutOps[i], corePortsInValues[i+1]).getResult());
        }
        
    } else { // Pull
        // Pull: CORES (Out) -> [MEM (In)] -> SHIM (Out)
        // Producers: all corePortOut, memPortIn (if mem exists)
        // Consumers: shimPortOut (for shim sends to DDR)
        
        if (memTile) { // CORES -> MEM -> SHIM
            hops.push_back(rewriter.create<dmaphop::create_hop>(loc, memPortOut, shimPortIn).getResult());
            hops.push_back(rewriter.create<dmaphop::create_hop>(loc, corePortsOutOps.back(), memPortIn).getResult());
            // Producers: all core output ports, memPortIn (mem receives from cores)
            for (size_t i = 0; i < corePortsOutOps.size(); ++i) {
                std::string outPortName = "corePortOut" + std::to_string(i);
                producerPortSymbols.push_back(SymbolRefAttr::get(rewriter.getContext(), outPortName));
            }
            producerPortSymbols.push_back(SymbolRefAttr::get(rewriter.getContext(), "memPortIn"));
            // Consumer: shimPortOut (shim sends to DDR)
            consumerPortSymbols.push_back(SymbolRefAttr::get(rewriter.getContext(), "shimPortOut"));
        } else { // CORES -> SHIM
            hops.push_back(rewriter.create<dmaphop::create_hop>(loc, corePortsOutOps.back(), shimPortIn).getResult());
            // Producers: all core output ports
            for (size_t i = 0; i < corePortsOutOps.size(); ++i) {
                std::string outPortName = "corePortOut" + std::to_string(i);
                producerPortSymbols.push_back(SymbolRefAttr::get(rewriter.getContext(), outPortName));
            }
            // Consumer: shimPortOut (shim sends to DDR)
            consumerPortSymbols.push_back(SymbolRefAttr::get(rewriter.getContext(), "shimPortOut"));
        }
        
        // Create hops between cores (if multiple cores)
        for (int i = corePortsOutOps.size() - 1; i >= 1; --i) {
            hops.push_back(rewriter.create<dmaphop::create_hop>(loc, corePortsOutOps[i - 1], corePortsInValues[i]).getResult());
        }
    }

    // --- 7. Create path, buffers, and final data movement op ---
    mlir::MLIRContext *ctx = rewriter.getContext();
    ArrayAttr produceArray = rewriter.getArrayAttr(producerPortSymbols);
    mlir::ArrayAttr consumeArray = mlir::ArrayAttr::get(ctx, consumerPortSymbols);
    
    // For Push: shim/mem produces, cores consume
    // For Pull: cores produce, shim/mem consumes
    auto path = (direction == DataflowDirection::Push)
        ? rewriter.create<dmaphop::create_path>(loc, hops, produceArray, consumeArray, rewriter.getArrayAttr({}))
        : rewriter.create<dmaphop::create_path>(loc, hops, produceArray, consumeArray, rewriter.getArrayAttr({}));

    auto memrefType = MemRefType::get(ArrayRef<int64_t>{1024}, rewriter.getF32Type());
    Value ddrBuffer = rewriter.create<memref::AllocOp>(loc, memrefType);
    Value coreBufferTemplate = rewriter.create<memref::AllocOp>(loc, memrefType);
    SmallVector<Value, 4> coreBuffers;
    for (auto coreTile : coreTiles) {
        coreBuffers.push_back(rewriter.create<dmaphop::alloc_buffer>(loc, memrefType, coreTile.getResult(), coreBufferTemplate).getResult());
    }

    if (direction == DataflowDirection::Push) {
        rewriter.create<dmaphop::push>(loc, ddrBuffer, path, coreBuffers, corePortsInValues);
    } else {
        rewriter.create<dmaphop::pull>(loc, ddrBuffer, path, coreBuffers, corePortsInValues);
    }
    rewriter.create<dmaphop::sync>(loc, path);

    // --- 8. Deallocate buffers and erase original op ---
    for (auto buffer : coreBuffers) {
        rewriter.create<dmaphop::dealloc_buffer>(loc, buffer);
    }
    rewriter.create<memref::DeallocOp>(loc, ddrBuffer);
    rewriter.create<memref::DeallocOp>(loc, coreBufferTemplate);
    rewriter.eraseOp(op);
    return success();
}

// Lowering for dmap::push. This is now a thin wrapper.
struct PushOpLowering : public OpConversionPattern<dmap::push> {
    explicit PushOpLowering(MLIRContext *context, RoutingTopology &router)
        : OpConversionPattern<dmap::push>(context), router_(router) {}

    LogicalResult matchAndRewrite(dmap::push op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        return lowerDataMovementOp(op, rewriter, router_, DataflowDirection::Push);
    }
private:
    RoutingTopology &router_;
};

// Lowering for dmap::pull. This is now a thin wrapper.
struct PullOpLowering : public OpConversionPattern<dmap::pull> {
    explicit PullOpLowering(MLIRContext *context, RoutingTopology &router)
        : OpConversionPattern<dmap::pull>(context), router_(router) {}

    LogicalResult matchAndRewrite(dmap::pull op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        return lowerDataMovementOp(op, rewriter, router_, DataflowDirection::Pull);
    }
private:
    RoutingTopology &router_;
};

// Generic lowering pattern to erase an op that is no longer needed.
template <typename Op_T>
struct EraseOpLowering : public OpConversionPattern<Op_T> {
    using OpConversionPattern<Op_T>::OpConversionPattern;
    LogicalResult matchAndRewrite(Op_T op, typename Op_T::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return success();
    }
};

} // namespace

void DmapToDmaphopPass::runOnOperation() {
    auto& ctx = getContext();
    ConversionTarget target(ctx);
    RewritePatternSet patterns(&ctx);

    // The conversion is successful when the dmap dialect is gone.
    target.addIllegalDialect<dmap::dmapdialect>();
    // These dialects are legal to have in the output.
    target.addLegalDialect<dmaphop::dmaphopdialect, func::FuncDialect, memref::MemRefDialect>();

    // Add the primary lowering patterns.
    //patterns.add<DmapFuncOpLowering, PushOpLowering>(&ctx);
    patterns.add<PushOpLowering>(&ctx, rtopology_);
    patterns.add<PullOpLowering>(&ctx, rtopology_);
    
    // Add patterns to erase the old dmap ops that are now handled by the main patterns.
    patterns.add<
        EraseOpLowering<dmap::create_data>,
        EraseOpLowering<dmap::define_io_engine>,
        EraseOpLowering<dmap::define_core_group>,
        EraseOpLowering<dmap::define_port_configure>,
        EraseOpLowering<dmap::create_io_engin_with_config>,
        EraseOpLowering<dmap::create_core_group_with_config>,
        EraseOpLowering<dmap::createstream>,
        EraseOpLowering<dmap::createchainstream>//,
        //EraseOpLowering<dmap::push>
    >(&ctx);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

DmapToDmaphopPass::DmapToDmaphopPass(RoutingTopology& rtopology):rtopology_(rtopology) {
}