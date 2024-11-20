#ifndef SSTREE_H
#define SSTREE_H

#include <vector>
#include <limits>
#include <algorithm>
#include <numeric>
#include "Point.h"
#include "Data.h"
#include <queue>
#include "utils.h"

class SSNode {
    size_t maxPointsPerNode;
    Point centroid;
    float radius;
    float rmin;
    SSNode* parent;
    std::vector<SSNode*> children;
    std::vector<Data*> _data;

    // For searching
    SSNode* findClosestChild(const Point& target);

    // For insertion
    size_t directionOfMaxVariance();
    std::pair<SSNode*, SSNode*> split();
    size_t findSplitIndex(size_t coordinateIndex);
    [[nodiscard]] std::vector<Point> getEntriesCentroids() const;
    size_t minVarianceSplit(const std::vector<float>& values);

public:
    bool isLeaf;
    SSNode(const Point& centroid,size_t maxPointsPerNode, float radius=0.0f, bool isLeaf=true, SSNode* parent=nullptr)
        : centroid(centroid),maxPointsPerNode(maxPointsPerNode), radius(radius), isLeaf(isLeaf), parent(parent) {
        rmin = std::numeric_limits<float>::max();
    }

    // Checks if a point is inside the bounding sphere
    bool intersectsPoint(const Point& point) const;

    void updateBoundingEnvelope();

    // Getters
    const Point& getCentroid() const { return centroid; }
    float getRadius() const { return radius; }
    const std::vector<SSNode*>& getChildren() const { return children; }
    const std::vector<Data  *>& getData    () const { return    _data; }
    bool getIsLeaf() const { return isLeaf; }
    SSNode* getParent() const { return parent; }

    // Insertion
    SSNode* searchParentLeaf(SSNode* node, const Point& target);
    std::pair<SSNode*, SSNode*> insert(SSNode*& node, Data* data);
    void setParent(SSNode* _parent){parent = _parent;}

    // Search
    SSNode *search(SSNode *node, Data *_data);

    //Variance
    float varianceAlongDirection(const std::vector<Point> & centroids, const size_t & dimension);
    static float getMean(const std::vector<Point> &centroids, const size_t &dimension);
    float getMean(const std::vector<float> &values);
    float getVariance(const std::vector<float> &values);

    //
    void insertNode(SSNode * node);
    void knn(const Point& q, const size_t& k, std::priority_queue<Data*, std::vector<Data*>, QueryComparator>& L, float& Dk);
};

class SSTree {
private:
    SSNode* root;
    size_t maxPointsPerNode;

public:
    SSTree(size_t maxPointsPerNode)
        : maxPointsPerNode(maxPointsPerNode), root(nullptr) {}

    void insert(Data* _data);
    SSNode* search(Data* _data) const;

    SSNode * getRoot() const;
    std::vector<Data*> knn(const Point & query, const size_t & k) const;
};

#endif // SSTREE_H