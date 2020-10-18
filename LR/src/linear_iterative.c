#include <math.h>
#include "linear.h"

static
void r_squared(
  linear_param_t * params,
  data_t * dataset, 
  sum_t * sumset, 
  result_t * response) 
{
  float mean = sumset->sumy / params->size;
  rsquared_t dist = {0};
  float y_estimated = 0;

  for (int i = 0; i < params->size; i++) {
    dist.actual += pow((dataset[i].y - mean), 2);
    y_estimated = dataset[i].x * response->a1 + response->a0;
    dist.estimated += pow((y_estimated - mean), 2);
  }

  // LOG_RSQUARED_T(dist);
  response->rsquared = dist.estimated / dist.actual * 100;
}

void iterative_regression(
  linear_param_t * params, 
  data_t * dataset, 
  result_t * response) 
{
  START_TIME
  sum_t sumset = {0};

  for (int i = 0; i < params->size; i++) {
    sumset.sumx   += dataset[i].x;
    sumset.sumy   += dataset[i].y;
    sumset.sumxsq += pow(dataset[i].x, 2);
    sumset.sumxy  += dataset[i].x * dataset[i].y;
  }
  
  double det = params->size * sumset.sumxsq - pow(sumset.sumx, 2);

  response->a0 = (sumset.sumy * sumset.sumxsq - sumset.sumx * sumset.sumxy) / det;
  response->a1 = (params->size * sumset.sumxy - sumset.sumx * sumset.sumy) / det;

  r_squared(params, dataset, &sumset, response);

  END_TIME
  response->time = CALC_TIME;
}
