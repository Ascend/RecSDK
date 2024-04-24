//
// Created by z00576261 on 2024/4/15.
//

#include "mapper_fast.h"
#include <cmath>
#include <cstring>
#include <pthread.h>
#include <iostream>

RecMapper::BuckStatus RecMapper::InnerBuck::Insert(uint64_t key, uint64_t& value, std::function<bool()> ValueSet)
{
    for (int i = 0; i < BUCKCAPACITY; ++i){
        uint64_t old_key = 0;
        if (keys_[i].load(std::memory_order_relaxed) == 0 && keys_[i].compare_exchange_strong(old_key, key)){
            bool ret = ValueSet();
            if (!ret){
                keys_[i].store(0);
                return BuckStatus::BUCK_ERROR;
            }
            values_[i] = value;
            return BuckStatus::BUCK_NOEXIST;
        }
    }
    return BuckStatus::BUCK_ERROR;
}

RecMapper::BuckStatus RecMapper::InnerBuck::Find(uint64_t key, uint64_t& value)
{
    for (int i = 0; i < BUCKCAPACITY; ++i){
        if (keys_[i].load(std::memory_order_relaxed) == key){
            value = values_[i];
            return BuckStatus::BUCK_EXIST;
        }
    }
    return BuckStatus::BUCK_NOEXIST;
}

RecMapper::BuckStatus RecMapper::InnerBuck::Remove(uint64_t key)
{
    for (int i = 0; i < BUCKCAPACITY; ++i) {
        uint64_t oldkey = key;
        if (keys_[i].load(std::memory_order_relaxed) == key){
            if (keys_[i].compare_exchange_strong(oldkey, 0)){
                values_[i] = 0;
                return BuckStatus::BUCK_EXIST;
            }
        }
    }
    return BUCK_ERROR;
}

bool RecMapper::MapperFast::InitializeBuck()
{
    uint16_t i = 0;

    while(i <= prime_max){
        if (pow(2, i) < reserve_){
            i++;
            continue;
        }
        break;
    }
    buck_count_ = i < 7 ? 128 : pow(2, i);

    for(auto &buck_map : buck_maps_){
        InnerBuck* buck_map_temp = new (std::nothrow) InnerBuck[buck_count_];
        if (buck_map_temp == nullptr) {
            FreeBuckMaps();
            return false;
        }
        memset(buck_map_temp, 0, sizeof(InnerBuck) * buck_count_);
        buck_map = buck_map_temp;
    }
    return true;
}

void RecMapper::MapperFast::UnInitializeBuck()
{
    FreeBuckExpend();
    FreeBuckMaps();
}

void RecMapper::MapperFast::FreeBuckMaps()
{
    for (auto &buck_map : buck_maps_){
        if (buck_map != nullptr){
            delete[] buck_map;
            buck_map = nullptr;
        }
    }
}

void RecMapper::MapperFast::FreeBuckExpend()
{
    for (auto &buck_map : buck_maps_ ){
        if (buck_map == nullptr){
            continue;
        }
        for (uint32_t i = 0; i < buck_count_; ++i){
            InnerBuck* buck_attch = buck_map[i].next_;
            while (buck_attch != nullptr){
                InnerBuck* buck_attch_temp = buck_attch->next_;
                delete buck_attch;
                buck_attch = buck_attch_temp;
            }
        }
    }
}

RecMapper::MapperStatus RecMapper::MapperFast::Put(uint64_t key, uint64_t& value)
{
    if (size_.load() > capacity_){
        return MapperStatus::MAPPER_ERROR;
    }

    if(key == 0){
        if (spec_buck != nullptr) {
            spec_buck->spin.lock();
            spec_buck->Find(key, value);
            spec_buck->spin.unlock();
            return MapperStatus::MAPPER_OK;
        }
        spec_buck =  new (std::nothrow) InnerBuck;
        memset(spec_buck, 0, sizeof(InnerBuck));
        spec_buck->spin.lock();
        spec_buck->keys_[0].store(key);
        spec_buck->values_[0] = offset_.fetch_add(1) + 1;
        size_.fetch_add(1);
        spec_buck->spin.unlock();
        return MapperStatus::MAPPER_OK;
    }
    InnerBuck* buck = &(buck_maps_[key % sub_map_count][key % buck_count_]);
    //first，find key if exist in buck
    while(buck != nullptr){
        buck->spin.lock();
        if(buck->Find(key, value) == BuckStatus::BUCK_EXIST){
            buck->spin.unlock();
            return MapperStatus::MAPPER_OK;
        }
        buck->spin.unlock();
        if(buck->next_ != nullptr){
            buck = buck->next_;
        } else{
            break;
        }
    }

    //if not find,
    for (int i = 0; i < 8192; ++i){
        // insert exist buck
        while(buck != nullptr){
            buck->spin.lock();
            auto value_func = [&]() ->bool {
                value = offset_.fetch_add(1);
                return true;};
            BuckStatus ret = buck->Insert(key, value, value_func);

            buck->spin.unlock();
            if (ret == BuckStatus::BUCK_ERROR) {
                return MapperStatus::MAPPER_ERROR;
            } else if (ret == BuckStatus::BUCK_NOEXIST) {
                size_.fetch_add(1);
                return MapperStatus::MAPPER_OK;
            }
            if (buck->next_ != nullptr) {
                buck = buck->next_;
            } else {
                break;
            }
        }

        //insert not exist buck
        auto& old_spin = buck->spin;
        old_spin.lock();
        if (buck->next_ != nullptr) {
            buck = buck->next_;
            old_spin.unlock();
            continue;
        }

        InnerBuck* new_buck =  new (std::nothrow) InnerBuck;
        memset(new_buck, 0, sizeof(InnerBuck));
        buck->next_ = new_buck;
        buck = new_buck;
        old_spin.unlock();
    }
    return MapperStatus::MAPPER_ERROR;
}

RecMapper::MapperStatus RecMapper::MapperFast::Find(uint64_t key, uint64_t& value) {
    if(key == 0) {
        if (spec_buck != nullptr) {
            spec_buck->spin.lock();
            value = spec_buck->values_[0];
            spec_buck->spin.unlock();
            return MapperStatus::MAPPER_OK;
        }
        return MapperStatus::MAPPER_INVALID;
    }
    InnerBuck* buck = &(buck_maps_[key % sub_map_count][key % buck_count_]);
    if (buck == nullptr) {
        return MapperStatus::MAPPER_ERROR;
    }
    if (buck->Find(key,value) == BuckStatus::BUCK_NOEXIST) {
        return  MapperStatus::MAPPER_INVALID;
    }
    return MapperStatus::MAPPER_OK;
}

RecMapper::MapperStatus RecMapper::MapperFast::Remove(uint64_t key)
{
    if(key == 0) {
        if (spec_buck != nullptr) {
            delete spec_buck;
            spec_buck = nullptr;
            size_.fetch_sub(1);
            return MapperStatus::MAPPER_OK;
        }
        return MapperStatus::MAPPER_INVALID;
    }
    InnerBuck* buck = &(buck_maps_[key % sub_map_count][key % buck_count_]);
    while(buck != nullptr) {
        uint64_t value;
        if (buck->Find(key, value) == BuckStatus::BUCK_NOEXIST) {
            return MapperStatus::MAPPER_INVALID;
        }

        buck->spin.lock();
        if (buck->Remove(key) == BuckStatus::BUCK_EXIST){
            size_.fetch_sub(1);
            return MapperStatus::MAPPER_OK;
        }
        buck = buck->next_;
    }
    return MapperStatus::MAPPER_INVALID;
}

RecMapper::MapperStatus RecMapper::MapperFast::ToVector(std::vector<std::pair<uint64_t, uint64_t>>& vec)
{
    if (spec_buck != nullptr) {
        vec.push_back(std::make_pair<uint64_t, uint64_t>(spec_buck->keys_[0], spec_buck->values_[0]));
    }
    for (auto& sub_map : buck_maps_){
        if (sub_map == nullptr){
            continue;
        }
        for(int i = 0; i < buck_count_; ++i){
            InnerBuck* buck = &sub_map[i];
            while(buck) {
                for (int j = 0; j < BUCKCAPACITY; ++j){
                    if (buck->keys_[j] == 0) {
                        continue;
                    }
                    vec.push_back(std::make_pair<uint64_t, uint64_t>(buck->keys_[j], buck->values_[j]));
                }
                buck = buck->next_;
            }
        }
    }
    return MapperStatus::MAPPER_OK;
}