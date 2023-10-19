/*
 * DynamicDimensionQueue.h
 *
 *  Created on: Feb 14, 2023
 *      Author: mizolotu
 */

#ifndef DYNAMICDIMENSIONQUEUE_H_
#define DYNAMICDIMENSIONQUEUE_H_

//#include <vector>

class DynamicDimensionQueue {

  float *arr;          // array to store queue elements

  short dimension = 0;     // dimension of the queue
  short capacity = 0;      // maximum capacity of the queue
  short front = 0;         // front points to the front element in the queue (if any)
  short rear = 0;          // rear points to the last element in the queue
  short count = 0;         // current size of the queue

  float *sum;          // sum of elements

  float *ssum;         // sum of element squares

  private:

  public:

    DynamicDimensionQueue();
    DynamicDimensionQueue(short cap);
    DynamicDimensionQueue(short cap, short dim);

    ~DynamicDimensionQueue();

    void enqueue(float x);        // add an element to the queue
    void enqueue(float *x);       // add an element to the queue
    void dequeue();               // remove the first element
    short size();                 // get the queue size
    bool isEmpty();               // check whether the queue is empty
    bool isFull();                // check whether the queue is full
    float xmax();                 // calculate the queue max
    void xmax(float* m);          // calculate the queue max
    float mean();                 // calculate the queue mean
    void mean(float* m);          // calculate the queue mean
    float std();                  // calculate the queue std
    void std(float* s);           // calculate the queue std
    float get(short i);           // get an element of the queue
    void get(short i, float* x);  // get an element of the queue
    void clear();                 // clear the queue
};

#endif /* DYNAMICDIMENSIONQUEUE_H_ */
