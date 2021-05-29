__kernel void numeric_reduction(
  __global  int * data, 
  __local   int * interns,
  __global  int * result) 
{
  size_t glob_id = get_global_id(0);
  size_t loc_id = get_local_id(0);
  size_t loc_size = get_local_size(0); 

  interns[loc_id] = data[glob_id];

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = loc_size / 2 ; i > 0 ; i /= 2)
  {
    if (loc_id <= i) {
      // Only first half of workitems on each workgroup
      interns[loc_id] += interns[loc_id + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (loc_id == 0) {
    result[get_group_id(0)] = interns[0];
  }
}
