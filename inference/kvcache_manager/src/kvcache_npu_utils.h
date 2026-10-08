#pragma once

#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <cstdio>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace kvcache {

// ACL 调用错误检查宏：抛出 runtime_error（包含文件名、行号、ACL 错误消息）
#define ACL_CHECK(ans)                                                                                            \
    do {                                                                                                          \
        aclError _err = (ans);                                                                                    \
        if (_err != ACL_SUCCESS) {                                                                                \
            const char* msg = aclGetRecentErrMsg();                                                               \
            throw std::runtime_error(std::string("ACL error ") + std::to_string(_err) + " at " + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + ": " + (msg ? msg : "(no message)"));             \
        }                                                                                                         \
    } while (0)

// ACL 调用错误检查宏（noexcept 版本）：用于析构函数等不可抛异常的上下文，仅打印 stderr
#define ACL_CHECK_NOEXCEPT(ans)                                                                             \
    do {                                                                                                    \
        aclError _err = (ans);                                                                              \
        if (_err != ACL_SUCCESS) {                                                                          \
            const char* msg = aclGetRecentErrMsg();                                                         \
            std::fprintf(stderr, "ACL error %d at %s:%d: %s\n", static_cast<int>(_err), __FILE__, __LINE__, \
                         (msg ? msg : "(no message)"));                                                     \
        }                                                                                                   \
    } while (0)

// RAII 包装：aclrtStream，析构自动 aclrtDestroyStream，禁拷贝支持移动
class ACLStreamPtr {
public:
    ACLStreamPtr() noexcept = default;
    explicit ACLStreamPtr(aclrtStream s) noexcept : s_(s) {}
    ACLStreamPtr(const ACLStreamPtr&) = delete;
    ACLStreamPtr& operator=(const ACLStreamPtr&) = delete;
    ACLStreamPtr(ACLStreamPtr&& o) noexcept : s_(o.s_)
    {
        o.s_ = nullptr;
    }
    ACLStreamPtr& operator=(ACLStreamPtr&& o) noexcept
    {
        if (this != &o) {
            reset();
            s_ = o.s_;
            o.s_ = nullptr;
        }
        return *this;
    }
    ~ACLStreamPtr()
    {
        reset();
    }
    void reset(aclrtStream s = nullptr) noexcept
    {
        if (s_ != s) {
            if (s_) {
                ACL_CHECK_NOEXCEPT(aclrtDestroyStream(s_));
            }
            s_ = s;
        }
    }
    aclrtStream get() const noexcept
    {
        return s_;
    }
    explicit operator bool() const noexcept
    {
        return s_ != nullptr;
    }

private:
    aclrtStream s_ = nullptr;
};

// RAII 包装：aclrtEvent，析构自动 aclrtDestroyEvent
class ACLEventPtr {
public:
    ACLEventPtr() noexcept = default;
    explicit ACLEventPtr(aclrtEvent e) noexcept : e_(e) {}
    ACLEventPtr(const ACLEventPtr&) = delete;
    ACLEventPtr& operator=(const ACLEventPtr&) = delete;
    ACLEventPtr(ACLEventPtr&& o) noexcept : e_(o.e_)
    {
        o.e_ = nullptr;
    }
    ACLEventPtr& operator=(ACLEventPtr&& o) noexcept
    {
        if (this != &o) {
            reset();
            e_ = o.e_;
            o.e_ = nullptr;
        }
        return *this;
    }
    ~ACLEventPtr()
    {
        reset();
    }
    void reset(aclrtEvent e = nullptr) noexcept
    {
        if (e_ != e) {
            if (e_) {
                ACL_CHECK_NOEXCEPT(aclrtDestroyEvent(e_));
            }
            e_ = e;
        }
    }
    aclrtEvent get() const noexcept
    {
        return e_;
    }
    explicit operator bool() const noexcept
    {
        return e_ != nullptr;
    }

private:
    aclrtEvent e_ = nullptr;
};

// RAII 包装：Host pinned memory (aclrtMallocHost)，析构自动 aclrtFreeHost
class ACLHostMemPtr {
public:
    ACLHostMemPtr() noexcept = default;
    explicit ACLHostMemPtr(void* p) noexcept : p_(p) {}
    ACLHostMemPtr(const ACLHostMemPtr&) = delete;
    ACLHostMemPtr& operator=(const ACLHostMemPtr&) = delete;
    ACLHostMemPtr(ACLHostMemPtr&& o) noexcept : p_(o.p_)
    {
        o.p_ = nullptr;
    }
    ACLHostMemPtr& operator=(ACLHostMemPtr&& o) noexcept
    {
        if (this != &o) {
            reset();
            p_ = o.p_;
            o.p_ = nullptr;
        }
        return *this;
    }
    ~ACLHostMemPtr()
    {
        reset();
    }
    void reset(void* p = nullptr) noexcept
    {
        if (p_ != p) {
            if (p_) {
                ACL_CHECK_NOEXCEPT(aclrtFreeHost(p_));
            }
            p_ = p;
        }
    }
    void* get() const noexcept
    {
        return p_;
    }
    explicit operator bool() const noexcept
    {
        return p_ != nullptr;
    }

private:
    void* p_ = nullptr;
};

inline ACLHostMemPtr make_acl_host_mem(size_t bytes)
{
    void* p = nullptr;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&p), bytes));
    return ACLHostMemPtr(p);
}

inline ACLStreamPtr make_acl_stream()
{
    aclrtStream s = nullptr;
    ACL_CHECK(aclrtCreateStream(&s));
    return ACLStreamPtr(s);
}

inline ACLEventPtr make_acl_event()
{
    aclrtEvent e = nullptr;
    ACL_CHECK(aclrtCreateEvent(&e));
    return ACLEventPtr(e);
}

}  // namespace kvcache
