#include <stdio.h>

typedef struct {
  int x;
  int y;
} point_t;

point_t a = { 3, 2 };

void print_point(point_t * pt) {
  printf("(%d, %d)\n", pt->x, pt->y);
}

int main() {
  point_t b[32] = { {1} };
  // b = a;
  // b.y = 56;
  print_point(&b[1]);
}
