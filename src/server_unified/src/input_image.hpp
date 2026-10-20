#pragma once

#include <string>

namespace server {

// The edit worker's input: an uploaded PNG, JPEG, BMP or GIF, center-cropped to a
// square and resized to 1024x1024 RGB. The lossless PNG written for the worker is removed when
// this object is destroyed. Throws BadRequest for images that cannot be decoded.
class WorkerInputImage {
public:
    explicit WorkerInputImage(const std::string& encoded);
    ~WorkerInputImage();
    WorkerInputImage(const WorkerInputImage&) = delete;
    WorkerInputImage& operator=(const WorkerInputImage&) = delete;

    const std::string& path() const { return path_; }

private:
    std::string path_;
};

}  // namespace server
