//
// Created by rivas on 19/11/2024.
//

#ifndef UTILS_H
#define UTILS_H
#include "Data.h"

class QueryComparator {
    const Point& query;

public:
    QueryComparator(const Point& query) : query(query) {}

    bool operator()(const Data* data1, const Data* data2) const {
        return Point::distance(query, data1->getEmbedding()) <
               Point::distance(query, data2->getEmbedding());
    }
};
#endif //UTILS_H
