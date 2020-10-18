__kernel void array_2d(__global int *data)
{
  size_t id = get_global_id(1) * get_global_size(0) + get_global_id(0);
  printf("(%d;%d) = %d\n", get_global_id(0), get_global_id(1), data[id]);
  data[id] *= 2; 
}
