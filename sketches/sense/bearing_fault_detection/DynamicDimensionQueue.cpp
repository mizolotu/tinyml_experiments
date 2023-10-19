/*
 * DynamicDimensionQueue.cpp
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#include <math.h>

#include "DynamicDimensionQueue.h"
#include "utils.h"

//#include <iostream>
//using namespace std;

DynamicDimensionQueue::DynamicDimensionQueue() {}

DynamicDimensionQueue::DynamicDimensionQueue(short cap) {

  dimension = 1;

  arr = new float[cap];
  //

  for (short i=0; i < cap; i++) {
    arr[i] = 0.0;
    //arr.push_back(0.0);
  }
  capacity = cap;
  front = 0;
  rear = -1;
  count = 0;

  sum = new float[1];
  //
  ssum = new float[1];
  //

  for (short i=0; i < 1; i++) {

    sum[i] = 0.0;
    //sum.push_back(0.0);

    ssum[i] = 0.0;
    //ssum.push_back(0.0);

  }

}

DynamicDimensionQueue::DynamicDimensionQueue(short cap, short dim) {

  dimension = dim;

  arr = new float[cap * dim];
  //

  for (short i=0; i < cap * dim; i++) {

    arr[i] = 0.0;
    //arr.push_back(0.0);

  }
  capacity = cap;
  front = 0;
  rear = -1;
  count = 0;

  sum = new float[dim];
  //
  ssum = new float[dim];
  //

  for (short i=0; i < dim; i++) {

    sum[i] = 0.0;
    //sum.push_back(0.0);

    ssum[i] = 0.0;
    //ssum.push_back(0.0);
  }

}

DynamicDimensionQueue::~DynamicDimensionQueue() {
  //delete[] arr;
}

void DynamicDimensionQueue::dequeue() {

  for (short i=0; i < dimension; i++) {
    sum[i] -= arr[front * dimension + i];
    //cout << "minus this: " << arr[front * dimension + i] << endl;
    ssum[i] -= pow(arr[front * dimension + i], 2);
  }

  front = (front + 1) % capacity;
    count--;

}

void DynamicDimensionQueue::enqueue(float x) {

  //cout << "\nenqueued: " << x << endl << endl;

    if (isFull()) {
      //cout << "dequeued" << endl;
        dequeue();
    }

    rear = (rear + 1) % capacity;

    arr[rear] = x;
    //arr.at(rear) = x;

    sum[0] = sum[0] + x;
    //sum.at(0) = sum.at(0) + x;

    ssum[0] = ssum[0] + pow(x, 2);
    //ssum.at(0) = ssum.at(0) + pow(x, 2);

    count++;

    /*
    cout << "arr = ";
    for (short i=0; i < capacity * dimension; i++) {
      cout << arr[i] << ",";
    }
    cout << endl;
    */

}


void DynamicDimensionQueue::enqueue(float *x) {

    if (isFull()) {
        dequeue();
    }

    rear = (rear + 1) % capacity;

    for (short i=0; i < dimension; i++) {
      arr[rear * dimension + i] = x[i];
      sum[i] += x[i];
      ssum[i] += pow(x[i], 2);
    }

    count++;
}

short DynamicDimensionQueue::size() {
    return count;
}

bool DynamicDimensionQueue::isEmpty() {
    return (size() == 0);
}

bool DynamicDimensionQueue::isFull() {
    return (size() == capacity);
}

float DynamicDimensionQueue::xmax() {
  float m = -99999999.0;
  for (short j = 0; j < count; j++) {
    //cout << "j = " << j << ", arr_j = " << arr[j] << endl;
    if (arr[j] > m) {
      m = arr[j];
    }
  }
  return m;
}

void DynamicDimensionQueue::xmax(float* m) {
  for (short i = 0; i < dimension; i++) {
    m[i] = -99999999.0;
    for (short j = 0; j < count; j++) {
      if (arr[j * dimension + i] > m[i]) {
        m[i] = arr[j * dimension + i];
      }
    }
  }
}

float DynamicDimensionQueue::mean() {
  //cout << sum[0] << "," << count << endl;
  return sum[0] / count;
}

void DynamicDimensionQueue::mean(float* m) {
  for (short i = 0; i < dimension; i++) {
    m[i] = sum[i] / count;
  }
}

float DynamicDimensionQueue::std() {
  return sqrt(maximum(0.0, (ssum[0] - count * pow(sum[0] / count, 2)) / count));
}

void DynamicDimensionQueue::std(float *s) {
  for (short i = 0; i < dimension; i++) {
    s[i] = sqrt(maximum(0.0, (ssum[i] - count * pow(sum[i] / count, 2)) / count));
  }
}

float DynamicDimensionQueue::get(short i) {
  return arr[(i + front) % capacity];
}

void DynamicDimensionQueue::get(short i, float* x) {
  for (short j = 0; j < dimension; j++) {
    x[j] = arr[((i + front) % capacity) * dimension + j];
  }
}

void DynamicDimensionQueue::clear() {
  for (short i=0; i < capacity * dimension; i++) {
    arr[i] = 0.0;
  }
  for (short i=0; i < dimension; i++) {
    sum[i] = 0.0;
    ssum[i] = 0.0;
  }
  front = 0;
  rear = -1;
  count = 0;

  /*
  cout << "after clear, arr = ";
  for (short i=0; i < capacity * dimension; i++) {
      cout << arr[i] << ",";
  }
  cout << endl;
  */

}
