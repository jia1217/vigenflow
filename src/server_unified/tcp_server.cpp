#include "tcp_server.hpp"

#include "http_handlers.hpp"
#include "server_config.hpp"

#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <filesystem>
#include <iostream>
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
    tcp::acceptor acceptor{ioc, {net::ip::make_address("0.0.0.0"), config::port}};

    std::cout << "Server running on http://0.0.0.0:" << config::port << "\n";
    std::cout << "Default Model: " << config::default_model_id << "\n";
    for (const auto& model : config::models) {
        std::cout << "Model ID: " << model.id << "\n"
                  << "  run.exe: " << model.worker_exe << "\n"
                  << "  Weights Path: " << model.weights_path << "\n"
                  << "  NPU Files Path: " << model.npu_files_path << "\n"
                  << "  Supported steps: " << config::supported_steps_text(model) << "\n";
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
