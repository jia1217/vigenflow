// CPU regression checks; no model weights or NPU required.
// From src/server_webui:
// g++ -std=c++17 -O2 -Isrc tests/edit_sizes_test.cpp src/input_image.cpp src/request_parser.cpp \
//   src/server_config.cpp src/lora_list.cpp src/file_utils.cpp -lpng -o /tmp/edit_sizes_test
#include "input_image.hpp"
#include "request_parser.hpp"
#include "server_config.hpp"
#include <png.h>
#include <nlohmann/json.hpp>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
using nlohmann::json;

std::string fixture(int width, int height, bool horizontal) {
    std::vector<unsigned char> rgb(width * height * 3);
    for (int y=0; y<height; ++y) for (int x=0; x<width; ++x) {
        const int position=horizontal?x:y, extent=horizontal?width:height;
        // Colored outer quarters make an incorrect crop immediately visible.
        rgb[(y*width+x)*3+(position<extent/4?0:position>=extent*3/4?2:1)]=255;
    }
    png_image image{}; image.version=PNG_IMAGE_VERSION; image.width=width;
    image.height=height; image.format=PNG_FORMAT_RGB;
    png_alloc_size_t length=0;
    assert(png_image_write_to_memory(&image,nullptr,&length,0,rgb.data(),0,nullptr));
    std::string data(length,'\0');
    assert(png_image_write_to_memory(&image,data.data(),&length,0,rgb.data(),0,nullptr));
    png_image_free(&image); data.resize(length); return data;
}

void crop_check(int src_width, int src_height, bool horizontal, int width, int height) {
    std::string path;
    {
        server::WorkerInputImage input(fixture(src_width,src_height,horizontal),width,height);
        path=input.path();
        png_image image{}; image.version=PNG_IMAGE_VERSION;
        assert(png_image_begin_read_from_file(&image,path.c_str()));
        assert(image.width==width && image.height==height);
        image.format=PNG_FORMAT_RGB;
        std::vector<unsigned char> rgb(PNG_IMAGE_SIZE(image));
        assert(png_image_finish_read(&image,nullptr,rgb.data(),0,nullptr));
        for (int y : {height/4,height/2,height*3/4}) for (int x : {width/4,width/2,width*3/4}) {
            assert(rgb[(y*width+x)*3]==0);
            assert(rgb[(y*width+x)*3+1]==255);
            assert(rgb[(y*width+x)*3+2]==0);
        }
        png_image_free(&image);
    }
    assert(!std::filesystem::exists(path));
}

int main() {
    for (const auto& model : server::config::models) {
        if (!model.edits_image) continue;
        json body={{"model",model.id},{"prompt","Change the color."},{"loras","none"}};
        for (const char* size : {"768x1024","1024x768","848x1488","16x2048","2048x16","2048x2048","auto"}) {
            body["size"]=size;
            const auto request=server::parse_edit_request(body,"image bytes");
            const auto expected=std::string(size)=="auto"?"1024x1024":size;
            assert(std::to_string(request.params.width)+"x"+std::to_string(request.params.height)==expected);
        }
        body.erase("size");
        assert(server::parse_edit_request(body,"image bytes").params.width==1024);
        for (const json& size : json::array({"0x768","15x768","769x1024","2064x768","768x-16","768X1024","768x1024x16",1024,nullptr})) {
            body["size"]=size; bool rejected=false;
            try { server::parse_edit_request(body,"image bytes"); }
            catch(const server::BadRequest&) { rejected=true; }
            assert(rejected);
        }
    }
    crop_check(300,600,false,1024,1024); // Existing square behavior.
    crop_check(600,300,true,256,512);   // Landscape source to portrait.
    crop_check(300,600,false,512,256);  // Portrait source to landscape.
    bool rejected=false;
    try { server::WorkerInputImage input("not an image",768,1024); }
    catch(const server::BadRequest&) { rejected=true; }
    assert(rejected);
    std::cout << "PASS: edit dimensions, invalid sizes, defaults, crop pixels, PNG dimensions and temporary-file cleanup.\n";
}
