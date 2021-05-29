__kernel void array(
  __global int* data, 
  __global int* out_data) 
{
  // printf("%d->%d:%d\n", get_global_id(0), get_local_id(0), get_local_size(0));
  out_data[get_global_id(0)] = data[get_global_id(0)] * 2;
}
