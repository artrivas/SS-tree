#include "SSTree.h"

#include <vector>

constexpr float INF = std::numeric_limits<float>::max();




/**
 * intersectsPoint
 * Verifica si un punto está dentro de la esfera delimitadora del nodo.
 * @param point: Punto a verificar.
 * @return bool: Retorna true si el punto está dentro de la esfera, de lo contrario false.
 */
bool SSNode::intersectsPoint(const Point& point) const {
    return Point::distance(centroid, point) <= radius;
}

/**
 * findClosestChild
 * Encuentra el hijo más cercano a un punto dado.
 * @param target: El punto objetivo para encontrar el hijo más cercano.
 * @return SSNode*: Retorna un puntero al hijo más cercano.
 */
SSNode* SSNode::findClosestChild(const Point& target) {
    assert(!isLeaf && "A leaf node\n");
    float minDist = INF;
    SSNode* result = nullptr;
    for(SSNode* & childNode : children) {
        if(childNode && Point::distance(childNode->centroid,target) < minDist) {
            minDist = Point::distance(childNode->centroid, target);
            result = childNode;
        }
    }
    return result;
}

/**
 * updateBoundingEnvelope
 * Actualiza el centroide y el radio del nodo basándose en los nodos internos o datos.
 */

float SSNode::getMean(const std::vector<Point> &centroids, const size_t &dimension) {
    float mean = 0;
    for(const Point & centroid : centroids) {
        mean += centroid[dimension];
    }
    mean/=static_cast<float>(centroids.size());
    return mean;
}

float SSNode::getMean(const std::vector<float> &values) {
    float mean = 0;
    for(const float value : values) {
        mean += value;
    }
    mean/=static_cast<float>(values.size());
    return mean;
}

void SSNode::updateBoundingEnvelope() {
    const std::vector<Point> points = this->getEntriesCentroids();

    Eigen::VectorXf centroidCoordinates = Eigen::VectorXf::Zero(DIM);

    for (const auto& centroid : points) {
        centroidCoordinates += centroid.coordinates_;
    }

    centroidCoordinates /= points.size();
    centroid = Point(centroidCoordinates);

    if(this->isLeaf) {
        for(const Data * entry : this->_data) {
            float dist = Point::distance(this->centroid,entry->getEmbedding());
            this->radius = std::max(this->radius,dist);
            this->rmin   = std::min(this->rmin,dist);
        }
    }else {
        for(SSNode* &entry : this->children) {
            float dist = Point::distance(this->centroid,entry->getCentroid());
            this->radius = std::max(this->radius,dist+entry->getRadius());
            this->rmin = std::min(this->rmin,dist);
        }
    }

}


float SSNode::getVariance(const std::vector<float> &values) {
    const float mean = getMean(values);
    float variance = 0;
    for(const float value : values)
        variance+=(value - mean)*(value-mean);
    variance/=static_cast<float>(values.size());
    return variance;
}

float SSNode::varianceAlongDirection(const std::vector<Point> &centroids, const size_t &dimension) {
    const float mean = getMean(centroids, dimension);
    float variance = 0;
    for(const Point & centroid : centroids)
        variance+=(centroid[dimension] - mean)*(centroid[dimension]-mean);
    variance/=static_cast<float>(centroids.size());
    return variance;
}

/**
 * directionOfMaxVariance
 * Calcula y retorna el índice de la dirección de máxima varianza.
 * @return size_t: Índice de la dirección de máxima varianza.
 */
size_t SSNode::directionOfMaxVariance() {
    float maxVariance = 0;
    size_t directionIndex = 0;
    const std::vector<Point> centroids = this->getEntriesCentroids();
    for(size_t i = 0; i < DIM; ++i) {
        if(varianceAlongDirection(centroids,i)>maxVariance) {
            maxVariance = varianceAlongDirection(centroids,i);
            directionIndex = i;
        }
    }
    return directionIndex;
}

/**
 * split
 * Divide el nodo y retorna el nuevo nodo creado.
 * Implementación similar a R-tree.
 * @return SSNode*: Puntero al nuevo nodo creado por la división.
 */
std::pair<SSNode*, SSNode*> SSNode::split() {
    const size_t splitIndex = findSplitIndex(directionOfMaxVariance());
    SSNode* newNode1 = new SSNode(centroid, maxPointsPerNode, radius, isLeaf, this->parent);
    SSNode* newNode2 = new SSNode(centroid, maxPointsPerNode, radius, isLeaf, this->parent);

    if (isLeaf) {
        newNode1->_data = std::vector<Data*>(_data.begin(), _data.begin() + static_cast<int>(splitIndex));
        newNode2->_data = std::vector<Data*>(_data.begin() + static_cast<int>(splitIndex), _data.end());;
    } else {
        newNode1->children = std::vector<SSNode*>(children.begin(), children.begin() + static_cast<int>(splitIndex));
        newNode2->children = std::vector<SSNode*>(children.begin() + static_cast<int>(splitIndex), children.end());
    }

    newNode1->updateBoundingEnvelope();
    newNode2->updateBoundingEnvelope();


    this->updateBoundingEnvelope();
    this->isLeaf = false;

    return {newNode1, newNode2};
}


/**
 * findSplitIndex
 * Encuentra el índice de división en una coordenada específica.
 * @param coordinateIndex: Índice de la coordenada para encontrar el índice de división.
 * @return size_t: Índice de la división.
 */
size_t SSNode::findSplitIndex(size_t coordinateIndex) {
    if(this->isLeaf) {
        std::ranges::sort(this->_data,[coordinateIndex](const Data* a, const Data* b)  {
            return a->getEmbedding()[coordinateIndex] < b->getEmbedding()[coordinateIndex];
        });
    }else {
        std::ranges::sort(this->children,[coordinateIndex](const SSNode* a, const SSNode* b)  {
            return a->getCentroid()[coordinateIndex] < b->getCentroid()[coordinateIndex];
        });
    }
    std::vector<float> points;
    for(const Point &point : this->getEntriesCentroids())
        points.emplace_back(point[coordinateIndex]);
    return minVarianceSplit(points);
}

/**
 * getEntriesCentroids
 * Devuelve los centroides de las entradas.
 * Estos centroides pueden ser puntos almacenados en las hojas o los centroides de los nodos hijos en los nodos internos.
 * @return std::vector<Point>: Vector de centroides de las entradas.
 */
std::vector<Point> SSNode::getEntriesCentroids() const {
    std::vector<Point> points;
    if(this->isLeaf) {
        for(const Data* i : this->_data) {
            points.emplace_back(i->getEmbedding());
        }
        return points;
    }
    for(const SSNode* child:this->children) {
        points.emplace_back(child->getCentroid());
    }
    return points;
}


/**
 * minVarianceSplit
 * Encuentra el índice de división óptimo para una lista de valores, de tal manera que la suma de las varianzas de las dos particiones resultantes sea mínima.
 * @param values: Vector de valores para encontrar el índice de mínima varianza.
 * @return size_t: Índice de mínima varianza.
 */
size_t SSNode::minVarianceSplit(const std::vector<float>& values) {
    float minVariance = INF;
    const size_t m = this->maxPointsPerNode/2; //Checkear
    size_t splitIndex = m; //No encontre el minimo
    for(size_t i = m; i < values.size()-m; ++i) {
        std::vector<float> v1(values.begin(),values.begin()+static_cast<int>(i)-1);
        std::vector<float> v2(values.begin()+static_cast<int>(i),values.end());
        const float variance2 = getVariance(v2);
        if(const float variance1 = getVariance(v1); variance1+variance2< minVariance) {
            minVariance = variance1+variance2;
            splitIndex = i;
        }
    }
    return splitIndex;
}

/**
 * searchParentLeaf
 * Busca el nodo hoja adecuado para insertar un punto.
 * @param node: Nodo desde el cual comenzar la búsqueda.
 * @param target: Punto objetivo para la búsqueda.
 * @return SSNode*: Nodo hoja adecuado para la inserción.
 */
SSNode* SSNode::searchParentLeaf(SSNode* node, const Point& target) {
    return (node->isLeaf ? node: searchParentLeaf(node->findClosestChild(target),target));
}

/**
 * insert
 * Inserta un dato en el nodo, dividiéndolo si es necesario.
 * @param node: Nodo donde se realizará la inserción.
 * @param data
 * @param _data: Dato a insertar.
 * @return SSNode*: Nuevo nodo raíz si se dividió, de lo contrario nullptr.
 */
std::pair<SSNode*, SSNode*> SSNode::insert(SSNode*& node, Data* data) {
    if (node->isLeaf) {
        for(const Data* point : node->_data) {
            if(point == data) {
                return {nullptr,nullptr};
            }
        }
        node->_data.emplace_back(data);
        node->updateBoundingEnvelope();

        if (node->_data.size() <= node->maxPointsPerNode) {
            return {nullptr, nullptr};
        }
    } else {
        SSNode* closestChild = node->findClosestChild(data->getEmbedding());
        auto [newChild1, newChild2] = insert(closestChild,data);
        if (newChild1 == nullptr) {
            node->updateBoundingEnvelope();
            return {nullptr, nullptr};
        }

        std::erase(node->children, closestChild);
        node->children.emplace_back(newChild1);
        node->children.emplace_back(newChild2);
        node->updateBoundingEnvelope();

        if (node->children.size() <= node->maxPointsPerNode) {
            return {nullptr, nullptr};
        }
    }
    return node->split();
}

/**
 * search
 * Busca un dato específico en el árbol.
 * @param node: Nodo desde el cual comenzar la búsqueda.
 * @param _data: Dato a buscar.
 * @return SSNode*: Nodo que contiene el dato (o nullptr si no se encuentra).
 */
SSNode* SSNode::search(SSNode* node, Data* _data) {
    if(node->isLeaf) {
        for(auto & point : node->getData()) {
            if(point == _data) {
                return node;
            }
        }
    }else {
        for(auto & childNode : node->getChildren()) {
            if(childNode->intersectsPoint(_data->getEmbedding())) {
                if(const auto result = search(childNode, _data)) { //Profe tal vez si no le corre es porque esta directiva es de c++20, creo ...
                    return result;
                }
            }
        }
    }
    return nullptr;
}

void SSNode::insertNode(SSNode * node) {
    this->children.emplace_back(node);
}

void SSNode::knn( const Point& query, const size_t & k, std::priority_queue<Data*, std::vector<Data*>, QueryComparator>& L, float& Dk) {
    if (this->isLeaf)
    {
        const float distQC = Point::distance(this->getCentroid(), query);
        for (const auto & o : this->_data) {
            if (const float distDC = Point::distance(o->getEmbedding(), this->getCentroid());
                distQC - distDC > Dk || distDC - distQC > Dk)
                continue;
            if (const float distQD = Point::distance(o->getEmbedding(), query); distQD < Dk) {
                L.push(o);
                if (L.size() > k)
                    L.pop();
                else if (L.size() == k)
                    Dk = Point::distance(L.top()->getEmbedding(), query);
            }
        }
    }
    else
    {
        std::ranges::sort(this->children,[&query](const SSNode* a, const SSNode* b) {
            return Point::distance(a->getCentroid(), query) < Point::distance(b->getCentroid(), query);
        });

        if (k == 1) {
            for (SSNode* child : this->children) {
                const float distC = Point::distance(child->getCentroid(), query);
                if (distC > Dk)
                    break;
                if (distC + child->rmin < Dk) {
                    Dk = distC + child->rmin;
                    continue;
                }
                child->knn(query, k, L, Dk);
            }
        }

        for (SSNode* child : this->children) {
            if (const float distC = Point::distance(child->getCentroid(), query); (distC - child->getRadius() <= Dk) && (child->rmin - distC <= Dk))
                child->knn(query, k, L, Dk);
        }

    }
}


/**
 * insert
 * Inserta un dato en el árbol.
 * @param _data: Dato a insertar.
 */
void SSTree::insert(Data* _data) {
    if (!root)
        root = new SSNode(_data->getEmbedding(), maxPointsPerNode);

    if (auto [newChild1, newChild2] = root->insert(root,_data); newChild1) {
        root = new SSNode(_data->getEmbedding(), maxPointsPerNode);

        root->insertNode(newChild1);
        root->insertNode(newChild2);
        root->isLeaf = false;
        root->updateBoundingEnvelope();
        newChild1->setParent(root);
        newChild2->setParent(root);
    }
}

/**
 * search
 * Busca un dato específico en el árbol.
 * @param _data: Dato a buscar.
 * @return SSNode*: Nodo que contiene el dato (o nullptr si no se encuentra).
 */
SSNode* SSTree::search(Data* _data) const {
    return root->search(root,_data);
}

SSNode * SSTree::getRoot() const {
    return this->root;
}

std::vector<Data *> SSTree::knn(const Point &query, const size_t &k) const {
    float Dk = std::numeric_limits<float>::max();
    const QueryComparator comparator(query);
    std::priority_queue<Data*, std::vector<Data*>, QueryComparator> L(comparator);
    if (!root) {
        return {};
    }
    root->knn(query, k, L, Dk);
    std::vector<Data*> result;
    while (!L.empty()) {
        result.emplace_back(L.top());
        L.pop();
    }
    std::ranges::reverse(result);
    return result;
}
