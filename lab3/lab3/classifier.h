#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include "image.h"
#include "matrix.h"
#include "vector.h"
#include "csc.h"

__constant__ extern int nc;
__constant__ extern Vector avgs[32];
__constant__ extern Matrix covs[32];

typedef struct
{
	int x;
	int y;
} Point;

typedef struct
{
	int size;
	Point* points;
} Class;

typedef struct
{
	int size;
	Class* classes;
	Vector* avgs;
	Matrix* covs;
} Classifier;

Classifier* createClassifier(Image* image);
void deleteClassifier(Classifier* classifier);
void copyClassifierToConstant(Classifier* classifier);

#endif