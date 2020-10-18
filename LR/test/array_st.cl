struct st {
  int x;
  int y;
};

__kernel void array_st(__global struct st* data) 
{
  size_t id = get_global_id(0);
  // data[id]++;
  // printf("%d: (%d,%d)\n", id, data[id].x, data[id].y);
  data.x = 0;
  data.y = 1;
}
