#ifndef TILING_POLICY_FACTORY
#define TILING_POLICY_FACTORY

#include "tiling_policy.h"

namespace HstuDenseForwardFuxi {

const int TYPE_NUM = 3;

enum class LAYOUT_TYPE {
    NORMAL = 0,
    NORMALV200 = 1,
    JAGGED = 2,
    INVALID = 3
};

class TilingPolicyFactory {
public:
    static std::shared_ptr<TilingPolicy> CreatePolicy(const char *layOutCStr);
    static void TilingPolicyRegister(LAYOUT_TYPE policyKey, std::shared_ptr<TilingPolicy> policy);
    static TilingPolicyFactory &GetInstance();
private:
    static LAYOUT_TYPE ParseLayout(const char *layOutCStr);
    static std::vector<std::shared_ptr<TilingPolicy>> m_policyMap;
};

class RegisterPolicy {
public:
    RegisterPolicy(LAYOUT_TYPE policyKey, std::shared_ptr<TilingPolicy> policy)
    {
        TilingPolicyFactory::GetInstance().TilingPolicyRegister(policyKey, policy);
    }
    ~RegisterPolicy()=default;
};

#define REGISTER_POLICY(policyKey, policy) \
    static RegisterPolicy g_##po(policyKey, policy);
}

#endif