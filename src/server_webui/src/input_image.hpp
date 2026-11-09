#pragma once

#include <string>

namespace server {

// The edit worker's input: an uploaded PNG, JPEG, BMP or GIF, center-cropped to a
// selected aspect ratio and resized to the requested RGB dimensions. The PNG is removed when
// this object is destroyed. Throws BadRequest for images that cannot be decoded.
class WorkerInputImage {
public:
    WorkerInputImage(const std::string& encoded, int width, int height);
    ~WorkerInputImage();
    WorkerInputImage(const WorkerInputImage&) = delete;
    WorkerInputImage& operator=(const WorkerInputImage&) = delete;

    const std::string& path() const { return path_; }

private:
    std::string path_;
};

}  // namespace server
