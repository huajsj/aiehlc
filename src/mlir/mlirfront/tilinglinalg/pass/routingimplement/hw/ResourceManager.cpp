/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

//====================================================================
// routingresource.cpp  —  ResourceMgr & RoutingTile implementation
//====================================================================
#include "hw/ResourceManager.h"
#include <iostream>
#include <algorithm> 
// ── global DataIO id init ──
std::atomic<int> DataIO::next_{0};

DataIO::DataIO(IOType tp, int r, int c, DMADIRECTION dir, int channel, std::string nm, std::string cmt)
    : id_(++next_), rowpos_(r), colpos_(c), type_(tp), dmaDirection_(dir), channel_(channel), name_(std::move(nm)), comment_(std::move(cmt)) {}
// ──────────────────────────────────────────────────────────────
// RoutingTile impl
// ──────────────────────────────────────────────────────────────
RoutingTile::RoutingTile(int r,int c, TileType tt,const std::vector<PortTemplate> & Portinfo)
    : row_(r), col_(c), type_(tt)
{
    for(const auto& tp : Portinfo){
        auto& vec = (tp.role==PortRole::Master)?
                     banks_[tp.dir].master : banks_[tp.dir].slave;
        vec.resize(tp.ports);
        // if there is an valid available_ports then update the vc.porNum
        for (int i = 0; i < std::min(tp.ports, (int)tp.available_ports.size()); i++) {
            vec[i].setportNum(tp.available_ports[i]);
        }
    }
    //std::cout << "routing construct done " << std::endl;
}

uint32_t RoutingTile::getPortnumFromPortIdx(PortDirection dir, PortRole role, uint32_t portidx)
{
     auto& vec = (role==PortRole::Master)? banks_[dir].master : banks_[dir].slave;
     //if vec is empty or the portNum is default -1, then port idx is the portnum
     if (vec.empty() || (-1 == vec[portidx].getportNum())) {
        return portidx;
     }
     return vec[portidx].getportNum();
}

std::optional<int> RoutingTile::occupyport(IOType io, PortDirection dir, int ioId){
    //the steam switch on the tile have master and slave port, when it output data
    // the slave port is used and connect to neighbor tile master, when input the
    //master port is the interface
    auto& vec = (io==IOType::Input)? banks_[dir].master : banks_[dir].slave;
    for (int i = 0; i < vec.size(); i++) {
        auto portidx = allocate(io, i, dir, ioId);
        if (portidx) {
            return portidx;
        }
    }
    return std::nullopt;
}
std::optional<int> RoutingTile::allocate(IOType io, int portidx, PortDirection dir, int ioId){
    //the steam switch on the tile have master and slave port, when it output data
    // the slave port is used and connect to neighbor tile master, when input the
    //master port is the interface
    auto& vec = (io==IOType::Input)? banks_[dir].master : banks_[dir].slave;
    //for(int ch=0; ch<(int)vec.size(); ++ch){
    if(!vec[portidx].used){ 
        vec[portidx].used=true;
        vec[portidx].invalid=false;
        vec[portidx].ioId = ioId;
        //keep portNum no change as the allocate only mark the port as used and should not change portnum
        return portidx; 
    }
    //}
    return std::nullopt;
}

bool RoutingTile::releaseByIo(IOType io, int portidx,  PortDirection dir, int ioId){
    auto& vec = (io==IOType::Input)? banks_[dir].slave : banks_[dir].master;
    //for(auto& slot: vec) 
    auto& slot = vec[portidx];
    if(slot.used && slot.ioId==ioId) {
        slot = {};
        return true;
    }
    return false;
}

// ──────────────────────────────────────────────────────────────
// ResourceMgr impl
// ──────────────────────────────────────────────────────────────
std::once_flag              ResourceMgr::flag_;
std::shared_ptr<ResourceMgr> ResourceMgr::singleton_;

ResourceMgr::ResourceMgr(std::unique_ptr<IHwResource> resource, TileType defType){
    int rows = resource->getRows();
    int cols = resource->getColumns();
    tiles_.resize(rows);
    for(int r=0;r<rows;++r){
        tiles_[r].reserve(cols);
        for(int c=0;c<cols;++c){
            TileType tt =resource->tileType(r, c);
            auto& portsinfo = resource->getPortsForTileType(tt);
            tiles_[r].emplace_back(r,c,tt,portsinfo);
        }
    }
    resource_ = std::move(resource);
    lastdioid = 0;
    InitSHIMNocList();
}

void ResourceMgr::InitSHIMNocList() {
    auto& shimnocs = resource_->getShimNoc();
    for (auto x:shimnocs) {
        addShimTile(std::make_shared<ShimTile>(0, x, 2, 2));
    }
}

void ResourceMgr::addShimTile(std::shared_ptr<ShimTile> shim) {
       TileCoord key{shim->row(), shim->col()};
       shimTiles_[key] = std::move(shim);
}

uint32_t ResourceMgr::allocdioid() {
    return ++lastdioid;
}

RoutingTile& ResourceMgr::tile(int r,int c){ return tiles_[r][c]; }
const RoutingTile& ResourceMgr::tile(int r,int c) const { return tiles_[r][c]; }

std::shared_ptr<DataIO> ResourceMgr::createDataIO(IOType tp, int r, int c, DMADIRECTION dir, int channel, std::string nm, std::string cmt) {
    std::shared_ptr<DataIO> dataioptr = std::make_shared<DataIO>(tp, r, c, dir, channel, nm, cmt);
    DataIOMap[dataioptr->id()] = dataioptr;
    TileType tt =resource_->tileType(r, c);
    if (tt == TileType::Shim) {
        RoutingTile& t = tile(r, c);
        auto portidx = t.occupyport(tp,PortDirection::South, dataioptr->id());
        if (portidx) {
            uint32_t portnum = t.getPortnumFromPortIdx(PortDirection::South, (tp == IOType::Input) ? PortRole::Slave : PortRole::Master, *portidx);
            auto shimport = std::make_optional<ShimIOPort>(tp,PortDirection::South,  portnum);
            dataioptr->setshimport(shimport);
        }
        
        // Register the shim column, channel, and direction to ioId mapping
        registerShimChannelMapping(c, channel, dir, dataioptr->id());
    }
    return dataioptr;
}
// ---------- linkAvailable ----------
//link a to link b means same port number of A slave and B master should both exist
bool ResourceMgr::linkAvailable(Point a, Point b, int& portIdx) const {
    PortDirection dir=getDir(a,b), odir=opposite(dir);
    const auto& va = tile(a.r,a.c).bank(dir).slave;
    const auto& vb = tile(b.r,b.c).bank(odir).master;
    int lim = std::min<int>(va.size(), vb.size());
    for(int ch=0; ch<lim; ++ch)
        if(!va[ch].used && !vb[ch].used){ portIdx=ch; return true; }
    return false;
}

bool ResourceMgr::portDirAvailable(Point a, int& portIdx, PortDirection direction, bool master) const {
    const auto& va = tile(a.r,a.c).bank(direction).slave;
    if (master) {
        tile(a.r,a.c).bank(direction).master;
    }
    int lim = va.size();
    for(int ch=0; ch<lim; ++ch)
        if(!va[ch].used ){ portIdx=ch; return true; }
    return false;
}

// ---------- occupyLink ----------
bool ResourceMgr::occupyLink(Point a, Point b,const int ioId,int& portidx, PortDirection& directionAtoB, PortDirection& directionBtoA) {
    int chosenPort; if(!linkAvailable(a,b,portidx)) return false;
    directionAtoB=getDir(a,b);
    directionBtoA=opposite(directionAtoB);
    if (tile(a.r,a.c).allocate(IOType::Output, portidx, directionAtoB , ioId) &&
        tile(b.r,b.c).allocate(IOType::Input, portidx, directionBtoA, ioId) ) {
            return true;
    }
    return false;
}

bool ResourceMgr::occupyPointDirection(Point a,int& portidx, PortDirection& pd, bool slave) {
    if (!portDirAvailable(a, portidx, pd, slave)) return false;
    if(tile(a.r,a.c).allocate(IOType::Output, portidx, pd , 0)) {
        return true;
    }
    return false;
}

// ---------- releaseLink (by ioId) ----------
bool ResourceMgr::releaseLink(Point a, Point b, int ioId,int portidx){
    PortDirection dir=getDir(a,b), odir=opposite(dir);
    bool ret = tile(a.r,a.c).releaseByIo(IOType::Output,portidx, dir , ioId);
    ret |= tile(b.r,b.c).releaseByIo(IOType::Input ,portidx, odir, ioId);
    return ret;
}

int ResourceMgr::rows() const {
    return resource_->getRows();
}
int ResourceMgr::cols() const {
    return resource_->getColumns();
}
//-------------------------
std::optional<FoundDmaSlot> ResourceMgr::freeShimNoc(std::optional<TypeBasedTileLoc> ioPaireddstTileloc,
                                                     DMADIRECTION direct,
                                                     int requesterIoId) const {
    auto iopaireddst = ioPaireddstTileloc->loc;
    std::shared_ptr<ShimTile> findShimTile;
    for (const auto& kv : shimTiles_) {
        ShimTile& t = *(kv.second);
        if (requesterIoId >= 0 && t.isReserved() && t.getReservedByIoId() != requesterIoId)
            continue;
        if ( t.hasAnyFreeChannelForEngine(direct)) {
            if (!findShimTile) {
                findShimTile = kv.second;
                continue;
            }
            auto c = t.col();
            uint32_t new_distance = std::abs((int)c - (int)iopaireddst.c);
            uint32_t old_distance = std::abs((int)findShimTile->col() - (int)iopaireddst.c);
            if (new_distance < old_distance) {
                findShimTile = kv.second;
                continue;
            } 
        }
    }

    if (findShimTile) {
        std::optional<int> ch = findShimTile->allocate(direct, -1 , requesterIoId);
        if (ch) {
           return std::make_optional <FoundDmaSlot> (FoundDmaSlot{direct, Point{0, findShimTile->col()}, static_cast <int> (*ch)});
        }
    }
    return  std::nullopt;
}
// ---------- freeShim 例子 ----------
std::optional<Point> ResourceMgr::freeShimNoc(std::optional<Point> dst) const {
    auto& shimnocs = resource_->getShimNoc();
    auto max_distance = resource_->getColumns();
    bool findAvailable = false;
    Point shimNoc = {0,-1};
    for(auto c:shimnocs){
        const RoutingTile& t = tile(0, c);
        if(t.type()!=TileType::Shim) continue;
        const auto& north = t.bank(PortDirection::North).slave;
        for (auto x:north) {
            if (!x.used) {
                //find the most distance close shim noc to the dst in col
                findAvailable = true;
                if (!dst) {
                    return Point{ 0, (int)c};
                } else {
                    uint32_t new_distance = std::abs((int)c - (int)dst->c);
                    uint32_t old_distance = std::abs((int)shimNoc.c - (int)dst->c);
                    if (shimNoc.c <0 || (new_distance < old_distance)) {
                        shimNoc = Point{ 0, (int)c};
                    }
                }
                
            }
        }
    }
    if (findAvailable) {
        return shimNoc;
    }
    return std::nullopt;
}

std::optional<Point> ResourceMgr::freeShimNoc(std::optional<TypeBasedTileLoc> loc) const {
    int row_offset = 0;
    if (!loc) {
        return std::nullopt;
    }
    
    Point abspos = loc->loc;
    switch(loc->ttype) {
        case TileType::Core:
            abspos.r = resource_->absTileRow(TileType::Core, loc->loc.r);
            break;
        case TileType::Mem:
            abspos.r = resource_->absTileRow(TileType::Mem, loc->loc.r);
            break;
        case TileType::Shim:
        case TileType::NocShim:
        case TileType::PLShim:
        case TileType::PLNocShim:
            abspos.r = resource_->absTileRow(TileType::Shim, loc->loc.r);
             break;
        case TileType::Unknown:
            std::cout << "the tile type is unknow check failed, force return" << std::endl;
            return std::nullopt;
    }
    std::cout << " relative row is " << loc->loc.r << " abs row is " << abspos.r << std::endl;
    auto absdstpoint = std::make_optional(abspos);
    return freeShimNoc(absdstpoint);
}

bool ResourceMgr::init(std::unique_ptr<IHwResource> resource, TileType defType)
{
  
    bool created = false;
    std::call_once(ResourceMgr::flag_, [&]{
        singleton_ = std::make_shared<ResourceMgr>(std::move(resource), defType);
        created = true;
    });
    return created;            // true=本次创建，false=之前已创建
}

std::shared_ptr<ResourceMgr> ResourceMgr::instance()
{
    if (!singleton_)
        throw std::runtime_error("ResourceMgr::init() has not been called");
    return singleton_;
}

// Check if a specific tile is reserved
bool ResourceMgr::isTileReserved(int r, int c) const {
    if (r < 0 || r >= rows() || c < 0 || c >= cols()) {
        return true; // Out of bounds tiles are considered reserved
    }
    return tile(r, c).isReserved();
}

// Reserve a specific tile for a DataIO
bool ResourceMgr::reserveTile(int r, int c, int ioId) {
    if (r < 0 || r >= rows() || c < 0 || c >= cols()) {
        return false; // Out of bounds
    }
    
    // Check if the tile is already reserved
    if (tile(r, c).isReserved()) {
        return false;
    }
    
    // Reserve the tile
    tile(r, c).setReserved(true, ioId);
    
    // Add this tile to the DataIO's reserved tiles list
    auto it = DataIOMap.find(ioId);
    if (it != DataIOMap.end()) {
        it->second->addReservedTile({r, c});
    }
    
    return true;
}

// Release all tiles reserved for a specific DataIO
void ResourceMgr::releaseReservedTiles(int ioId) {
    // Find all tiles reserved by this DataIO
    for (int r = 0; r < rows(); r++) {
        for (int c = 0; c < cols(); c++) {
            if (tile(r, c).isReserved() && tile(r, c).getReservedByIoId() == ioId) {
                tile(r, c).setReserved(false, -1);
            }
        }
    }
    
    // Clear the DataIO's reserved tiles list
    auto it = DataIOMap.find(ioId);
    if (it != DataIOMap.end()) {
        it->second->clearReservedTiles();
    }
}

// Get all tiles reserved by a specific DataIO
std::vector<Point> ResourceMgr::getReservedTilesForDataIo(int ioId) const {
    auto it = DataIOMap.find(ioId);
    if (it != DataIOMap.end()) {
        return it->second->getReservedTiles();
    }
    return {};
}

// Reserve multiple tiles for a DataIO using a strategy
// Reserve multiple tiles for a DataIO using a strategy with proximity consideration
// Original reserveTiles with startPoint, enhanced with type support
bool ResourceMgr::reserveTiles(int ioId, int numTiles, ReservationStrategy strategy, 
                              std::vector<Point>& allocatedTiles,
                              std::optional<TileType> requestedType,
                              std::optional<Point> startPoint) {
    allocatedTiles.clear();
    
    if (numTiles <= 0) {
        return true; // No tiles to allocate
    }
    
    // Get the DataIO's location
    auto it = DataIOMap.find(ioId);
    if (it == DataIOMap.end()) {
        return false; // Invalid DataIO ID
    }
    
    int dataIoRow = it->second->rowpos();
    int dataIoCol = it->second->colpos();
    
    // Override with startPoint if provided
    if (startPoint.has_value()) {
        dataIoRow = startPoint.value().r;
        dataIoCol = startPoint.value().c;
    }
    
    if (strategy == ReservationStrategy::COLUMN_FIRST) {
        // Create a vector of columns sorted by proximity to the DataIO column
        std::vector<int> colsByProximity;
        for (int c = 0; c < cols(); c++) {
            colsByProximity.push_back(c);
        }
        
        // Sort columns by distance to DataIO column
        std::sort(colsByProximity.begin(), colsByProximity.end(), 
            [dataIoCol](int a, int b) {
                return std::abs(a - dataIoCol) < std::abs(b - dataIoCol);
            });
        
        // Try each column in order of proximity to DataIO
        for (int c : colsByProximity) {
            int consecutiveAvailable = 0;
            int firstAvailableRow = -1;
            
            // Look for consecutive available tiles in this column
            for (int r = 0; r < rows(); r++) {
                // Check if tile is available and matches requested type if specified
                bool tileAvailable = !isTileReserved(r, c);
                if (requestedType.has_value()) {
                    tileAvailable = tileAvailable && (tile(r, c).type() == requestedType.value());
                }
                
                if (tileAvailable) {
                    if (consecutiveAvailable == 0) {
                        firstAvailableRow = r;
                    }
                    consecutiveAvailable++;
                    
                    // If we found enough tiles, reserve them
                    if (consecutiveAvailable == numTiles) {
                        // Temporarily store allocated tiles
                        std::vector<Point> tempAllocated;
                        
                        // Try to reserve all needed tiles
                        bool allReserved = true;
                        for (int i = 0; i < numTiles; i++) {
                            int row = firstAvailableRow + i;
                            Point p{row, c};
                            if (reserveTile(row, c, ioId)) {
                                tempAllocated.push_back(p);
                            } else {
                                allReserved = false;
                                break;
                            }
                        }
                        
                        if (allReserved) {
                            allocatedTiles = tempAllocated;
                            return true;
                        } else {
                            // Rollback partial allocations
                            for (const auto& p : tempAllocated) {
                                tile(p.r, p.c).setReserved(false, -1);
                            }
                            // Try next column
                            break;
                        }
                    }
                } else {
                    // Reset consecutive count when we hit a reserved/wrong type tile
                    consecutiveAvailable = 0;
                }
            }
        }
    } else { // ROW_FIRST strategy
        // Create a vector of rows sorted by proximity to the DataIO row
        std::vector<int> rowsByProximity;
        for (int r = 0; r < rows(); r++) {
            rowsByProximity.push_back(r);
        }
        
        // Sort rows by distance to DataIO row
        std::sort(rowsByProximity.begin(), rowsByProximity.end(), 
            [dataIoRow](int a, int b) {
                return std::abs(a - dataIoRow) < std::abs(b - dataIoRow);
            });
        
        // Try each row in order of proximity to DataIO
        for (int r : rowsByProximity) {
            int consecutiveAvailable = 0;
            int firstAvailableCol = -1;
            
            // Look for consecutive available tiles in this row
            for (int c = 0; c < cols(); c++) {
                // Check if tile is available and matches requested type if specified
                bool tileAvailable = !isTileReserved(r, c);
                if (requestedType.has_value()) {
                    tileAvailable = tileAvailable && (tile(r, c).type() == requestedType.value());
                }
                
                if (tileAvailable) {
                    if (consecutiveAvailable == 0) {
                        firstAvailableCol = c;
                    }
                    consecutiveAvailable++;
                    
                    // If we found enough tiles, reserve them
                    if (consecutiveAvailable == numTiles) {
                        // Temporarily store allocated tiles
                        std::vector<Point> tempAllocated;
                        
                        // Try to reserve all needed tiles
                        bool allReserved = true;
                        for (int i = 0; i < numTiles; i++) {
                            int col = firstAvailableCol + i;
                            Point p{r, col};
                            if (reserveTile(r, col, ioId)) {
                                tempAllocated.push_back(p);
                            } else {
                                allReserved = false;
                                break;
                            }
                        }
                        
                        if (allReserved) {
                            allocatedTiles = tempAllocated;
                            return true;
                        } else {
                            // Rollback partial allocations
                            for (const auto& p : tempAllocated) {
                                tile(p.r, p.c).setReserved(false, -1);
                            }
                            // Try next row
                            break;
                        }
                    }
                } else {
                    // Reset consecutive count when we hit a reserved/wrong type tile
                    consecutiveAvailable = 0;
                }
            }
        }
    }
    
    // If we get here, allocation failed
    return false;
}

// Register shim column, channel, and direction to ioId mapping
void ResourceMgr::registerShimChannelMapping(int shimCol, int channel, DMADIRECTION direction, int ioId) {
    shimChannelToIoIdMap_[std::make_tuple(shimCol, channel, direction)] = ioId;
}

// Find ioId based on shim column, channel number, and direction
std::optional<int> ResourceMgr::findIoIdByShimChannel(int shimCol, int channel, DMADIRECTION direction) const {
    auto key = std::make_tuple(shimCol, channel, direction);
    auto it = shimChannelToIoIdMap_.find(key);
    if (it != shimChannelToIoIdMap_.end()) {
        return it->second;
    }
    return std::nullopt;
}

// Find DataIO object by shim column, channel, and direction
std::shared_ptr<DataIO> ResourceMgr::findDataIOByShimChannel(int shimCol, int channel, DMADIRECTION direction) const {
    auto ioIdOpt = findIoIdByShimChannel(shimCol, channel, direction);
    if (ioIdOpt) {
        int ioId = *ioIdOpt;
        auto dioIt = DataIOMap.find(ioId);
        if (dioIt != DataIOMap.end()) {
            return dioIt->second;
        }
    }
    return nullptr;
}
