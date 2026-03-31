#ifndef __BUFFER_HPP__
#define __BUFFER_HPP__

#include <cstdint>
#include <cstddef>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <climits>

// Avoid including heavy non-standard headers in a header file.
// #include <bits/stdc++.h>
// #include <boost/program_options.hpp>

#define __XRT__

#ifdef __XRT__
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_device.h"
#endif

#include "debug_utils.hpp"

/*
This is a buffer wrapper that maps to a bo_buffer or other memory without performing a deep copy.
A copy (or mapping) does not duplicate the underlying memory; it only maps the pointer.
*/

// ===================== bytes class =====================
class bytes {
protected:
    std::unique_ptr<uint8_t[]> owned_data_;
    uint8_t* data_;
    size_t size_;
    bool is_owner_;
#ifdef __XRT__
    std::unique_ptr<xrt::bo> owned_bo_;
    xrt::bo* bo_;
    bool is_bo_owner_;
#endif

public:
    bytes() : data_(nullptr), size_(0), is_owner_(false)
#ifdef __XRT__
        , is_bo_owner_(false), bo_(nullptr), owned_bo_(nullptr)
#endif
    {}

    bytes(const bytes& other) : owned_data_(nullptr), data_(other.data_), size_(other.size_), is_owner_(false)
#ifdef __XRT__
        , is_bo_owner_(false), bo_(other.bo_), owned_bo_(nullptr)
#endif
    {}

    bytes(bytes&& other) noexcept
        : owned_data_(std::move(other.owned_data_)), data_(other.data_), size_(other.size_), is_owner_(other.is_owner_)
#ifdef __XRT__
        , is_bo_owner_(other.is_bo_owner_), bo_(other.bo_), owned_bo_(std::move(other.owned_bo_))
#endif
    {
        other.data_ = nullptr;
        other.size_ = 0;
        other.is_owner_ = false;
#ifdef __XRT__
        other.is_bo_owner_ = false;
        other.bo_ = nullptr;
        other.owned_bo_ = nullptr;
#endif
    }

    bytes(size_t size)
        : owned_data_(new uint8_t[size]), data_(owned_data_.get()), size_(size), is_owner_(true)
#ifdef __XRT__
        , is_bo_owner_(false), bo_(nullptr), owned_bo_(nullptr)
#endif
    {}

    bytes(uint8_t* data, size_t size)
        : owned_data_(nullptr), data_(data), size_(size), is_owner_(false)
#ifdef __XRT__
        , is_bo_owner_(false), bo_(nullptr), owned_bo_(nullptr)
#endif
    {}

#ifdef __XRT__
    bytes(xrt::bo& bo)
        : owned_data_(nullptr), data_(bo.map<uint8_t*>()), size_(bo.size()), is_owner_(false), is_bo_owner_(false), bo_(&bo), owned_bo_(nullptr)
    {}

    bytes(size_t size, xrt::device& device, xrt::kernel& kernel, int group_id, xrtBufferFlags flags = XRT_BO_FLAGS_HOST_ONLY)
        : owned_data_(nullptr), size_(size), is_owner_(false), is_bo_owner_(true)
    {   
        owned_bo_ = std::make_unique<xrt::bo>(device, size, flags, kernel.group_id(group_id));
        data_ = owned_bo_->map<uint8_t*>();
        bo_ = owned_bo_.get();
    }

    bytes(size_t size, void*user_ptr, xrt::device& device, xrt::kernel& kernel, int group_id, xrtBufferFlags flags = XRT_BO_FLAGS_HOST_ONLY)
        : owned_data_(nullptr), size_(size), is_owner_(false), is_bo_owner_(true)
    {   
        owned_bo_ = std::make_unique<xrt::bo>(device,user_ptr,  size, flags, kernel.group_id(group_id));
        data_ = owned_bo_->map<uint8_t*>();
        bo_ = owned_bo_.get();
    }    
#endif

    virtual ~bytes() {
        if (is_owner_) {
            data_ = nullptr;
            owned_data_.reset();
        }
#ifdef __XRT__
        if (is_bo_owner_) {
            bo_ = nullptr;
            owned_bo_.reset();
        }
#endif
    }

    bytes& operator=(const bytes& other) {
        if (this != &other) {
            owned_data_.reset();
            data_ = other.data_;
            size_ = other.size_;
            is_owner_ = false;
#ifdef __XRT__
            is_bo_owner_ = false;
            bo_ = other.bo_;
            owned_bo_.reset();
#endif
        }
        return *this;
    }

    bytes& operator=(bytes&& other) noexcept {
        if (this != &other) {
            // Release current resources before taking ownership of new ones
            owned_data_.reset();
            owned_data_ = std::move(other.owned_data_);
            data_ = other.data_;
            size_ = other.size_;
            is_owner_ = other.is_owner_;
#ifdef __XRT__
            owned_bo_.reset();
            is_bo_owner_ = other.is_bo_owner_;
            owned_bo_ = std::move(other.owned_bo_);
            bo_ = other.bo_;
            other.bo_ = nullptr;
            other.is_bo_owner_ = false;
#endif
            other.data_ = nullptr;
            other.size_ = 0;
            other.is_owner_ = false;
        }
        return *this;
    }

    uint8_t& operator[](size_t index) {
        assert(data_ && index < size_);
        return data_[index];
    }

    const uint8_t& operator[](size_t index) const {
        assert(data_ && index < size_);
        return data_[index];
    }

    size_t size() const { return size_; }
    uint8_t* data() const { return data_; }
    uint8_t* bdata() const { return data_; }
    uint8_t* begin() const { return data_; }
    uint8_t* end() const { return data_ + size_; }

    void copy_from(const uint8_t* src, size_t size) {
        assert(size <= size_);
        std::memcpy(data_, src, size);
    }

    void resize(size_t new_size) {
#ifdef __XRT__
        assert(!is_bo_owner_);
#endif
        if (data_ != nullptr && !is_owner_) {
            throw std::runtime_error("Cannot resize a non-owner buffer");
        }
        owned_data_.reset(new uint8_t[new_size]);
        data_ = owned_data_.get();
        size_ = new_size;
        is_owner_ = true;
    }

    void free() {
#ifdef __XRT__
        assert(!is_bo_owner_);
#endif
        owned_data_.reset();
        data_ = nullptr;
        size_ = 0;
        is_owner_ = false;
#ifdef __XRT__
        is_bo_owner_ = false;
        bo_ = nullptr;
        owned_bo_.reset();
#endif
    }

    void reserve(size_t size) { resize(size); }
    void release() { free(); }

    bool is_owner() const { return is_owner_; }
#ifdef __XRT__
    bool is_bo_owner() const { return is_bo_owner_; }
    void sync_to_device() { assert(bo_); bo_->sync(XCL_BO_SYNC_BO_TO_DEVICE); }
    void sync_from_device() { assert(bo_); bo_->sync(XCL_BO_SYNC_BO_FROM_DEVICE); }
    xrt::bo& bo() { assert(bo_); return *bo_; }
#endif

    void from_file(const std::string& filename, size_t offset = 0, size_t size = 0) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        if (size == 0) size = file_size;
        assert(size <= file_size);
        assert(offset + size <= size_);
        file.read(reinterpret_cast<char*>(data_) + offset, size);
        file.close();
    }
};
// --------------------- Derived class template: buffer<T> --------------------- //
// This class wraps a data type T over the underlying byte buffer.
template<typename T>
class buffer : public bytes {
public:
    // Default constructor.
    buffer() : bytes() {}

    // Construct a buffer for count elements (allocates memory).
    buffer(size_t count) : bytes(count * sizeof(T)) {}

    // Construct from existing T* data (shallow mapping; does not take ownership).
    buffer(T* data, size_t count)
        : bytes(reinterpret_cast<uint8_t*>(data), count * sizeof(T)) {}

    // Shallow copy constructor.
    buffer(const buffer& other) : bytes(other) {}

#ifdef __XRT__
    // Construct from an xrt::bo.
    buffer(xrt::bo& bo) : bytes(bo) {}

    // Construct a bo-backed buffer for count elements.
    buffer(size_t count, xrt::device& device, xrt::kernel& kernel, int group_id, xrtBufferFlags flags = XRT_BO_FLAGS_HOST_ONLY)
        : bytes(count * sizeof(T), device, kernel, group_id, flags) {}

    buffer(size_t count, T* user_buffer_ptr,  xrt::device& device, xrt::kernel& kernel, int group_id, xrtBufferFlags flags = XRT_BO_FLAGS_HOST_ONLY)
        : bytes(count * sizeof(T), (void*) user_buffer_ptr,  device, kernel, group_id, flags) {}        
#endif

    // --- New: Constructors from std::vector --- //

    // Construct from a const lvalue reference to a std::vector<T>.
    // This creates a shallow mapping (no deep copy); the caller must ensure that
    // the vector outlives this buffer.
    buffer(const std::vector<T>& vec)
        : bytes(reinterpret_cast<uint8_t*>(const_cast<T*>(vec.data())), vec.size() * sizeof(T))
    {
        // Ownership is not taken.
    }

    // Right-value (rvalue) constructor from std::vector<T>.
    // WARNING: This also creates a shallow mapping.
    // The caller must ensure that the vector is not used (and remains valid)
    // after constructing this buffer.
    buffer(std::vector<T>&& vec)
        : bytes(reinterpret_cast<uint8_t*>(vec.data()), vec.size() * sizeof(T))
    {
        // Do NOT attempt to adopt the vector's memory since std::vector does not provide a release() method.
        // Instead, document that the caller is responsible for maintaining the vector's lifetime.
    }

    // --- New: copy_from() overload that copies data from a std::vector --- //
    // Copies the contents of the vector into the buffer.
    // The buffer must have been allocated with the same number of elements.
    void copy_from(const std::vector<T>& vec) {
        if (vec.size() * sizeof(T) != this->size_) {
            throw std::runtime_error("Size mismatch in copy_from(vector)");
        }
        std::memcpy(data_, vec.data(), size_);
    }

    // --- Templated cast_to() --- //
    // Return a buffer<U> that reinterprets the underlying data as type U.
    //@warning This function is not safe, you should check the size of the buffer before calling this function
    template<typename U>
    buffer<U> cast_to() {
        // Calculate the number of U elements that can be mapped.
        size_t newCount = size_ / sizeof(U);
        return buffer<U>(reinterpret_cast<U*>(data_), newCount);
    }

    // Return a bytes view of this buffer (shallow copy).
    const bytes as_bytes() const {
        return *this;
    }
    // Move assignment operator: allows x = create_bo_buffer()
    buffer<T>& operator=(buffer<T>&& other) noexcept {
        bytes::operator=(std::move(other));
        return *this;
    }

    // Copy assignment operator (optional but good practice)
    buffer<T>& operator=(const buffer<T>& other) {
        bytes::operator=(other);
        return *this;
    }
    // Element access operators.
    T& operator[](size_t index) {
        assert(data_ != nullptr);
        assert(index < size());
        assert(index >= 0);
        return reinterpret_cast<T*>(data_)[index];
    }

    const T& operator[](size_t index) const {
        assert(data_ != nullptr);
        assert(index < size());
        assert(index >= 0);
        return reinterpret_cast<T*>(data_)[index];
    }

    // Return the number of T elements in the buffer.
    size_t size() const { return size_ / sizeof(T); }

    // Return pointer to T data.
    T* data() const { return reinterpret_cast<T*>(data_); }
    T* begin() const { return reinterpret_cast<T*>(data_); }
    T* end() const { return reinterpret_cast<T*>(data_) + size(); }

    // Resize the buffer to hold count elements (only allowed if not bo-mapped).
    void resize(size_t count) {
        bytes::resize(count * sizeof(T));
    }

    // Reserve capacity for count elements.
    void reserve(size_t count) {
        bytes::reserve(count * sizeof(T));
    }

    // Set all elements to the given value.
    void memset(T value) {
        T* ptr = data();
        for (size_t i = 0; i < size(); i++) {
            ptr[i] = value;
        }
    }

    // Copy data from another bytes object (must have the same byte size).
    void copy_from(const bytes& other) {
        if (size_ != other.size()) {
            throw std::runtime_error("Size mismatch in copy_from(bytes)");
        }
        memcpy(data_, other.data(), size_); // size_ is already in bytes.
    }

    // Copy data from another buffer (must have the same number of elements).
    void copy_from(const buffer<T>& other) {
        if (size() != other.size()) {
            throw std::runtime_error("Size mismatch in copy_from(buffer)");
        }
        memcpy(data_, other.bdata(), size_); // size_ is already in bytes.
    }

    // Copy data from a pointer and provided size.
    void copy_from(T* data, size_t size) {
        if (size > this->size()) {
            throw std::runtime_error("Size mismatch in copy_from(pointer)");
        }
        memcpy(data_, data, size * sizeof(T));
    }

    bytes& as_bytes() {
        return *this;
    }

    void from_file(const std::string& filename, size_t offset = 0, size_t size = 0) {
        bytes::from_file(filename, offset * sizeof(T), size * sizeof(T));
    }

};



#endif // __BUFFER_HPP__
