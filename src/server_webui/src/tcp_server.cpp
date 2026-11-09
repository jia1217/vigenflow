#include "tcp_server.hpp"

#include "http_handlers.hpp"
#include "lora_list.hpp"
#include "server_config.hpp"

#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;

namespace server {

namespace {

void do_session(tcp::socket socket) {
    beast::flat_buffer buffer;
    beast::error_code ec;

    for (;;) {
        http::request_parser<http::string_body> parser;
        parser.body_limit(52428800);

        http::read(socket, buffer, parser, ec);

        if (ec == http::error::end_of_stream) break;
        if (ec) {
            std::cerr << "read error: " << ec.message() << "\n";
            return;
        }

        http::request<http::string_body> request = parser.release();
        const bool keep_alive = request.keep_alive();

        beast::write(socket, handle_request(std::move(request)), ec);

        if (ec) {
            std::cerr << "write error: " << ec.message() << "\n";
            return;
        }
        if (!keep_alive) break;
    }

    socket.shutdown(tcp::socket::shutdown_send, ec);
}

}  // namespace

void run_server() {
    net::io_context ioc{1};
    tcp::acceptor acceptor{ioc};
    acceptor.open(tcp::v4());
    acceptor.set_option(tcp::acceptor::reuse_address(true));
    beast::error_code bind_error;
    acceptor.bind({net::ip::make_address("0.0.0.0"), config::port}, bind_error);
    if (bind_error == net::error::address_in_use) {
        throw std::runtime_error("port " + std::to_string(config::port) +
            " is already in use; choose another with --port PORT, or VGF_PORT=PORT ./start.sh");
    }
    if (bind_error) throw beast::system_error(bind_error, "bind");
    acceptor.listen();

    std::cout << "Server running on http://0.0.0.0:" << config::port << "\n";
    std::cout << "Browser UI: http://127.0.0.1:" << config::port << "/\n";
    std::cout << "Default Model: " << config::default_model_id << "\n";
    for (const auto& model : config::models) {
        if (!model.available) continue;
        std::cout << "Model ID: " << model.id << "\n"
                  << "  run.exe: " << model.worker_exe << "\n"
                  << "  Weights Path: " << model.weights_path << "\n"
                  << "  NPU Files Path: " << model.npu_files_path << "\n"
                  << "  Supported steps: " << config::supported_steps_text(model) << "\n"
                  << "  Sizes: "
                  << (model.edits_image ? std::to_string(config::kImageWidth) + "x" + std::to_string(config::kImageHeight)
                                        : "WIDTHxHEIGHT, " + std::to_string(config::kMinImageSide) + ".." +
                                              std::to_string(config::kMaxImageSide) + " in multiples of " +
                                              std::to_string(config::kImageSideMultiple))
                  << "\n";
        try {
            if (const auto loras = read_loras(model)) {
                std::cout << "  LoRAs (* = on by default):";
                for (const auto& lora : *loras) std::cout << " " << lora.name << (lora.enabled ? "*" : "");
                std::cout << (loras->empty() ? " none listed\n" : "\n");
            }
        } catch (const std::exception& e) {
            std::cout << "  LoRAs: " << e.what() << "\n";
        }
    }
    std::cout << "Image Output Dir: " << config::output_dir
              << (config::keep_images ? " (PNGs saved)" : " (temporary files removed after generation)") << "\n";

    if (!config::keep_images) {
        std::cout << "Image URLs: last " << config::kMemoryImageLimit << " images in memory; cleared on restart\n";
    }

    for (;;) {
        tcp::socket socket{ioc};
        acceptor.accept(socket);
        std::thread{do_session, std::move(socket)}.detach();
    }
}

}  // namespace server
