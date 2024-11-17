#include "SSTree.h"

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
    for(const auto& childNode : children) {
        if(childNode != nullptr && Point::distance(childNode->centroid,target) < minDist) {
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
    for(const auto & centroid : centroids) {
        mean += centroid[dimension];
    }
    mean/=static_cast<float>(centroids.size());
    return mean;
}

float SSNode::getMean(const std::vector<float> &values) {
    float mean = 0;
    for(size_t j = 0; j < values.size(); j++) {
        mean += values[j];
    }
    mean/=values.size();
    return mean;
}

void SSNode::updateBoundingEnvelope() {
    const auto points = this->getEntriesCentroids();
    for(size_t i = 0; i < DIM; ++i) {
        this->centroid[i] = getMean(points,i);
    }
    if(this->isLeaf) {
        for(const auto entry : this->_data) {
            this->radius = std::max(this->radius,Point::distance(this->centroid,entry->getEmbedding()));
        }
    }else {
        for(const auto entry : this->children) {
            if(entry)
                this->radius = std::max(this->radius,Point::distance(this->centroid,entry->getCentroid())+entry->getRadius());
        }
    }

}


float SSNode::getVariance(const std::vector<float> &values) {
    float mean = getMean(values);
    float variance = 0;
    for(size_t j = 0; j < values.size(); j++) {
        variance+=(values[j] - mean)*(values[j]-mean);
    }
    variance/=values.size();
    return variance;
}

float SSNode::varianceAlongDirection(const std::vector<Point> &centroids, const size_t &dimension) {
    float mean = getMean(centroids, dimension);
    float variance = 0;
    for(size_t j = 0; j < centroids.size(); j++) {
        variance+=(centroids[j][dimension] - mean)*(centroids[j][dimension]-mean);
    }
    variance/=centroids.size();
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
    const auto centroids = this->getEntriesCentroids();
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
SSNode* SSNode::split() {
    auto splitIndex = findSplitIndex(directionOfMaxVariance());
    SSNode* newNode = nullptr;
    if (isLeaf) {
        newNode = new SSNode(centroid, maxPointsPerNode, radius, true, this);
        newNode->_data = std::vector<Data*>(this->_data.begin()+splitIndex,this->_data.end());
        std::erase_if(this->_data,
                      [splitIndex, i = 0](const Data* d) mutable {
                          return i++ >= splitIndex;
                      });
    } else {
        newNode = new SSNode(centroid, maxPointsPerNode, radius, false, this);
        newNode->children = std::vector<SSNode*>(this->children.begin()+splitIndex,this->children.end());
        std::erase_if(this->_data,
                      [splitIndex, i = 0](const Data* d) mutable {
                          return i++ >= splitIndex;
                      });
    }
    this->updateBoundingEnvelope();
    newNode->updateBoundingEnvelope();
    return newNode;
}


/**
 * findSplitIndex
 * Encuentra el índice de división en una coordenada específica.
 * @param coordinateIndex: Índice de la coordenada para encontrar el índice de división.
 * @return size_t: Índice de la división.
 */
size_t SSNode::findSplitIndex(size_t coordinateIndex) {
    if(this->isLeaf) {
        std::sort(this->_data.begin(),this->_data.end(),[coordinateIndex](const Data* a, const Data* b)  {
            return a->getEmbedding()[coordinateIndex] < b->getEmbedding()[coordinateIndex];
        });
    }else {
        std::sort(this->children.begin(),this->children.end(),[coordinateIndex](const SSNode* a, const SSNode* b)  {
            return a->getCentroid()[coordinateIndex] < b->getCentroid()[coordinateIndex];
        });
    }
    std::vector<float> points;
    for(auto point : this->getEntriesCentroids()) {
        points.push_back(point[coordinateIndex]);
    }
    return minVarianceSplit(points);
}

/**
 * getEntriesCentroids
 * Devuelve los centroides de las entradas.
 * Estos centroides pueden ser puntos almacenados en las hojas o los centroides de los nodos hijos en los nodos internos.
 * @return std::vector<Point>: Vector de centroides de las entradas.
 */
std::vector<Point> SSNode::getEntriesCentroids() {
    std::vector<Point> points;
    if(this->isLeaf) {
        for(const auto& i : this->_data) {
            if(i)
                points.emplace_back(i->getEmbedding());
        }
        return points;
    }
    for(const auto& child:this->children) {
        if(child)
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
    size_t m = 2; //Checkear
    size_t splitIndex = m; //No encontre el minimo
    for(size_t i = m; i < values.size()-m; ++i) {
        std::vector<float> v1(values.begin(),values.begin()+i-1);
        std::vector<float> v2(values.begin()+i,values.end());
        float variance1 = getVariance(v1), variance2 = getVariance(v2);
        if(variance1+variance2< minVariance) {
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
    if(node->isLeaf)
        return node;
    return searchParentLeaf(node->findClosestChild(target), target);
}

/**
 * insert
 * Inserta un dato en el nodo, dividiéndolo si es necesario.
 * @param node: Nodo donde se realizará la inserción.
 * @param _data: Dato a insertar.
 * @return SSNode*: Nuevo nodo raíz si se dividió, de lo contrario nullptr.
 */
SSNode* SSNode::insert(SSNode* node, Data* data) {
    if (node->isLeaf) {
        for(const auto point: node->getData()) {
            if(point == data) {
                return nullptr;
            }
        }
        node->_data.push_back(data);
        node->updateBoundingEnvelope();
        if (node->_data.size() <= node->maxPointsPerNode) {
            return nullptr;
        }
    } else {
        SSNode* closestChild = node->findClosestChild(data->getEmbedding());
        auto newChild = node->insert(closestChild,data);
        if (!newChild) {
            node->updateBoundingEnvelope();
            return nullptr;
        }
        //
        auto temp = new SSNode(data->getEmbedding(),maxPointsPerNode); // creo un nuevo nodo;
        temp->insertNode(node);
        temp->insertNode(newChild);
        node = temp;
        node->isLeaf = false;
        //
        node->updateBoundingEnvelope();
        if (node->children.size() <= node->maxPointsPerNode) {
            return nullptr;
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


/**
 * insert
 * Inserta un dato en el árbol.
 * @param _data: Dato a insertar.
 */
void SSTree::insert(Data* data) {
    if (!root) {
        root = new SSNode(data->getEmbedding(), maxPointsPerNode);
    }

    auto newChild = root->insert(root,data);

    if (newChild) {
        auto temp = new SSNode(data->getEmbedding(),maxPointsPerNode); // creo un nuevo nodo;
        temp->insertNode(root);
        temp->insertNode(newChild);
        root = temp;
        root->isLeaf = false;
        root->updateBoundingEnvelope();
    }
}

/**
 * search
 * Busca un dato específico en el árbol.
 * @param _data: Dato a buscar.
 * @return SSNode*: Nodo que contiene el dato (o nullptr si no se encuentra).
 */
SSNode* SSTree::search(Data* _data) {
    return root->search(root,_data);
}

SSNode * SSTree::getRoot() const {
    return this->root;
}