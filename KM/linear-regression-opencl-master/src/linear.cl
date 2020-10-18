__kernel void linear_regression(
  __global  float2 * dataset,
  __local   float4 * interns,
  __global  float4 * result)
{
  size_t glob_id  = get_global_id(0);
  size_t loc_id   = get_local_id(0);
  size_t loc_size = get_local_size(0); 

  /* Initialize local buffer */
  interns[loc_id].s0 = dataset[glob_id].s0;
  interns[loc_id].s1 = dataset[glob_id].s1;
  interns[loc_id].s2 = (dataset[glob_id].s0 * dataset[glob_id].s1);
  interns[loc_id].s3 = (dataset[glob_id].s0 * dataset[glob_id].s0);
  
  barrier(CLK_LOCAL_MEM_FENCE);

  for (
    size_t  i = (loc_size / 2), old_i = loc_size; 
    i > 0; 
    old_i = i, i /= 2)
  {
    if (loc_id < i) {
      // Only first half of workitems on each workgroup
      interns[loc_id] += interns[loc_id + i];
      if (loc_id == (i - 1) && old_i % 2 != 0) {
        // If there is an odd number of data
        interns[loc_id] += interns[old_i - 1];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (loc_id == 0)
    result[get_group_id(0)] = interns[0];
}

__kernel void rsquared(
  __global float2 * dataset,
  float mean,
  float2 equation, // [a0,a1]
  __local float2 * dist,
  __global float2 * result)
{
  size_t glob_id  = get_global_id(0);
  size_t loc_id   = get_local_id(0);
  size_t loc_size = get_local_size(0); 

  dist[loc_id].s0 = pow((dataset[glob_id].s1 - mean), 2);

  float y_estimated = dataset[glob_id].s0 * equation.s1 + equation.s0;
  dist[loc_id].s1 = pow((y_estimated - mean), 2);

  barrier(CLK_LOCAL_MEM_FENCE);

  for (
    size_t  i = (loc_size / 2), old_i = loc_size; 
    i > 0; 
    old_i = i, i /= 2)
  {
    if (loc_id < i) {
      // Only first half of workitems on each workgroup
      dist[loc_id] += dist[loc_id + i];
      if (loc_id == (i - 1) && old_i % 2 != 0) {
        // If there is an odd number of data
        dist[loc_id] += dist[old_i - 1];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (loc_id == 0)
    result[get_group_id(0)] = dist[0];
}
