#ifndef __INFINIOP_EQ_API_H__
#define __INFINIOP_EQ_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopEqDescriptor_t;

__C __export infiniStatus_t infiniopCreateEqDescriptor(infiniopHandle_t handle,
                                                       infiniopEqDescriptor_t *desc_ptr,
                                                       infiniopTensorDescriptor_t c,
                                                       infiniopTensorDescriptor_t a,
                                                       infiniopTensorDescriptor_t b);

__C __export infiniStatus_t infiniopGetEqWorkspaceSize(infiniopEqDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopEq(infiniopEqDescriptor_t desc,
                                       void *workspace,
                                       size_t workspace_size,
                                       void *c,
                                       const void *a,
                                       const void *b,
                                       void *stream);

__C __export infiniStatus_t infiniopDestroyEqDescriptor(infiniopEqDescriptor_t desc);

#endif