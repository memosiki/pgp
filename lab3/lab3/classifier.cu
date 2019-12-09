#include "classifier.h"

__constant__ int nc;
__constant__ Vector avgs[32];
__constant__ Matrix covs[32];

Classifier* createClassifier(Image* image) {
    Classifier *c = (Classifier*)malloc(sizeof(Classifier));
    int nc, np;
    scanf("%d", &nc);
    c->avgs = (Vector*)malloc(sizeof(Vector) * nc);
    c->covs = (Matrix*)malloc(sizeof(Matrix) * nc);
    c->classes = (Class*)malloc(sizeof(Class) * nc);
    c->size = nc;
    for (int i = 0; i < nc; i++) {
        scanf("%d", &np);
        c->avgs[i].setAllToZero();
        c->covs[i].setAllToZero();
        c->classes[i].points = (Point *)malloc(sizeof(Point) * np);
        for (int j = 0; j < np; j++) {
            scanf("%d %d", &(c->classes[i].points[j].x), &(c->classes[i].points[j].y));
            Pixel p = getPixel(image, c->classes[i].points[j].x, c->classes[i].points[j].y);
            Vector tmp;
            tmp.createFromPixel(p);
            c->avgs[i].addVector(tmp);
        }
        c->avgs[i].multipleByNumber(1.0 / np);
        for (int j = 0; j < np; j++) {
            Pixel p = getPixel(image, c->classes[i].points[j].x, c->classes[i].points[j].y);
            Vector tmp;
            tmp.createFromPixel(p);
            tmp.subtractVector(c->avgs[i]);
            Matrix tmpMatrix = tmp.multipleByVectorTransposed(tmp);
            c->covs[i].addMatrix(tmpMatrix);
        }
        if (np > 1)
            c->covs[i].multipleByNumber(1.0 / (np - 1));
        c->covs[i] = c->covs[i].inverse();
    }
    return c;
}

void deleteClassifier(Classifier* c) {
    for (int i = 0; i < c->size; ++i)
		free(c->classes[i].points);
    free(c->avgs);
    free(c->covs);
    free(c->classes);
}

void copyClassifierToConstant(Classifier* c) {
    CSC(cudaMemcpyToSymbol(nc, &c->size, sizeof(c->size)));

	for (int i = 0; i < c->size; ++i)
	{
		CSC(cudaMemcpyToSymbol(avgs, &c->avgs[i], sizeof(Vector), i * sizeof(Vector)));
		CSC(cudaMemcpyToSymbol(covs, &c->covs[i], sizeof(Matrix), i * sizeof(Matrix)));
	}
}