#include "SSTree.h"

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
    return nullptr;
}

/**
 * updateBoundingEnvelope
 * Actualiza el centroide y el radio del nodo basándose en los nodos internos o datos.
 */
void SSNode::updateBoundingEnvelope() {
}

/**
 * directionOfMaxVariance
 * Calcula y retorna el índice de la dirección de máxima varianza.
 * @return size_t: Índice de la dirección de máxima varianza.
 */
size_t SSNode::directionOfMaxVariance() {
    return 0;
}

/**
 * split
 * Divide el nodo y retorna el nuevo nodo creado.
 * Implementación similar a R-tree.
 * @return SSNode*: Puntero al nuevo nodo creado por la división.
 */
SSNode* SSNode::split() {
    return nullptr;
}

/**
 * findSplitIndex
 * Encuentra el índice de división en una coordenada específica.
 * @param coordinateIndex: Índice de la coordenada para encontrar el índice de división.
 * @return size_t: Índice de la división.
 */
size_t SSNode::findSplitIndex(size_t coordinateIndex) {
    return 0;
}

/**
 * getEntriesCentroids
 * Devuelve los centroides de las entradas.
 * Estos centroides pueden ser puntos almacenados en las hojas o los centroides de los nodos hijos en los nodos internos.
 * @return std::vector<Point>: Vector de centroides de las entradas.
 */
std::vector<Point> SSNode::getEntriesCentroids() {
    return std::vector<Point>();
}


/**
 * minVarianceSplit
 * Encuentra el índice de división óptimo para una lista de valores, de tal manera que la suma de las varianzas de las dos particiones resultantes sea mínima.
 * @param values: Vector de valores para encontrar el índice de mínima varianza.
 * @return size_t: Índice de mínima varianza.
 */
size_t SSNode::minVarianceSplit(const std::vector<float>& values) {
    return 0;
}

/**
 * searchParentLeaf
 * Busca el nodo hoja adecuado para insertar un punto.
 * @param node: Nodo desde el cual comenzar la búsqueda.
 * @param target: Punto objetivo para la búsqueda.
 * @return SSNode*: Nodo hoja adecuado para la inserción.
 */
SSNode* SSNode::searchParentLeaf(SSNode* node, const Point& target) {
    return nullptr;
}

/**
 * insert
 * Inserta un dato en el nodo, dividiéndolo si es necesario.
 * @param node: Nodo donde se realizará la inserción.
 * @param _data: Dato a insertar.
 * @return SSNode*: Nuevo nodo raíz si se dividió, de lo contrario nullptr.
 */
SSNode* SSNode::insert(SSNode* node, Data* _data) {
    return nullptr;
}


/**
 * search
 * Busca un dato específico en el árbol.
 * @param node: Nodo desde el cual comenzar la búsqueda.
 * @param _data: Dato a buscar.
 * @return SSNode*: Nodo que contiene el dato (o nullptr si no se encuentra).
 */
SSNode* SSNode::search(SSNode* node, Data* _data) {
    return nullptr;
}


/**
 * insert
 * Inserta un dato en el árbol.
 * @param _data: Dato a insertar.
 */
void SSTree::insert(Data* _data) {
}


/**
 * search
 * Busca un dato específico en el árbol.
 * @param _data: Dato a buscar.
 * @return SSNode*: Nodo que contiene el dato (o nullptr si no se encuentra).
 */
SSNode* SSTree::search(Data* _data) {
    return nullptr;
}

SSNode * SSTree::getRoot() const {
    return this->root;
}
