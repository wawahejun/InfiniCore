#include "infiniccl_metax.h"

#include "../../utils.h"

#include <mccl.h>
#include <mcc/mcc_api.h>

#include <iostream>
#include <vector>

#define CHECK_HCCL(API__) CHECK_INTERNAL(API__, MCCL_SUCCESS)

inline mcc_stream_t getMacaStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return static_cast<mcc_stream_t>(stream);
}

inline mcclDataType_t getHcclDtype(infiniDtype_t datatype) {
    switch (datatype) {
    case INFINI_DTYPE_F32:
        return mcclFloat;
    case INFINI_DTYPE_F16:
        return mcclHalf;
    case INFINI_DTYPE_BF16:
        return mcclBfloat16;
    default:
        return mcclHalf;
    }
}

inline mcclRedOp_t getHcclRedOp(infinicclReduceOp_t op) {
    switch (op) {
    case INFINICCL_SUM:
        return mcclSum;
    case INFINICCL_PROD:
        return mcclProd;
    case INFINICCL_MAX:
        return mcclMax;
    case INFINICCL_MIN:
        return mcclMin;
    case INFINICCL_AVG:
        return mcclAvg;
    default:
        std::abort();
        return mcclSum;
    }
}

inline mcclComm_t getHcclComm(infinicclComm_t comm) {
    return static_cast<mcclComm_t>(comm->comm);
}

namespace infiniccl::metax {

infiniStatus_t commInitAll(
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids) {

    std::vector<mcclComm_t> hccl_comms(ndevice);
    CHECK_HCCL(mcclCommInitAll(hccl_comms.data(), ndevice, (int const *)device_ids));

    for (int i = 0; i < ndevice; i++) {
        comms[i] = new InfinicclComm{INFINI_DEVICE_METAX, device_ids[i], (void *)(hccl_comms[i])};
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t commDestroy(infinicclComm_t comm) {
    CHECK_HCCL(mcclCommDestroy(getHcclComm(comm)));
    delete comm;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t allReduce(
    void *sendbuf,
    void *recvbuf,
    size_t count,
    infiniDtype_t datatype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    CHECK_DTYPE(datatype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);

    CHECK_HCCL(mcclAllReduce(sendbuf, recvbuf, count, getHcclDtype(datatype),
                             getHcclRedOp(op), getHcclComm(comm), getMacaStream(stream)));

    return INFINI_STATUS_SUCCESS;
}
} // namespace infiniccl::metax
