#pragma once
// curl's binary pipe. Windows launches curl directly: cmd.exe would expand %VAR%
// in paths/URLs and interpret shell metacharacters. Tokens remain on stdin or in
// the inherited environment, never in the command line.
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/wait.h>
#endif

namespace prep {

inline std::string quote_curl_arg(const std::string& s) {
#ifdef _WIN32
    std::string q = "\"";
    size_t backslashes = 0;
    for (char c : s) {
        if (c == '\\') { ++backslashes; continue; }
        q.append(backslashes * (c == '"' ? 2 : 1), '\\');
        backslashes = 0;
        if (c == '"') q += '\\';
        q += c;
    }
    q.append(backslashes * 2, '\\');
    return q + '"';
#else
    std::string q = "'";
    for (char c : s) q += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return q + "'";
#endif
}

class CurlPipe {
public:
    CurlPipe(std::string command, bool read, const std::filesystem::path& stdout_file = {},
             const std::filesystem::path& stderr_file = {}) {
#ifdef _WIN32
        struct Handle {
            HANDLE value = INVALID_HANDLE_VALUE;
            ~Handle() { if (value != INVALID_HANDLE_VALUE && value) CloseHandle(value); }
        } parent, child, in, out, err;
        auto check = [](BOOL ok) {
            if (!ok) throw std::system_error(GetLastError(), std::system_category(), "starting curl.exe");
        };
        SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
        check(read ? CreatePipe(&parent.value, &child.value, &security, 0)
                   : CreatePipe(&child.value, &parent.value, &security, 0));
        check(SetHandleInformation(parent.value, HANDLE_FLAG_INHERIT, 0));
        auto open = [&](Handle& h, const std::filesystem::path& path, DWORD access, DWORD disposition) {
            h.value = CreateFileW(path.c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                  &security, disposition, FILE_ATTRIBUTE_NORMAL, nullptr);
            check(h.value != INVALID_HANDLE_VALUE);
        };
        auto standard = [&](Handle& h, DWORD id) {
            const HANDLE source = GetStdHandle(id);
            if (!source || source == INVALID_HANDLE_VALUE) open(h, L"NUL", GENERIC_WRITE, OPEN_EXISTING);
            else check(DuplicateHandle(GetCurrentProcess(), source, GetCurrentProcess(), &h.value,
                                       0, TRUE, DUPLICATE_SAME_ACCESS));
        };
        if (read) open(in, L"NUL", GENERIC_READ, OPEN_EXISTING);
        if (!read) {
            if (stdout_file.empty()) standard(out, STD_OUTPUT_HANDLE);
            else open(out, stdout_file, GENERIC_WRITE, CREATE_ALWAYS);
        }
        if (stderr_file.empty()) standard(err, STD_ERROR_HANDLE);
        else open(err, stderr_file, GENERIC_WRITE, CREATE_ALWAYS);

        STARTUPINFOEXW startup{};
        startup.StartupInfo.cb = sizeof(startup);
        startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        startup.StartupInfo.hStdInput = read ? in.value : child.value;
        startup.StartupInfo.hStdOutput = read ? child.value : out.value;
        startup.StartupInfo.hStdError = err.value;
        // Concurrent downloads must not inherit each other's pipe handles.
        HANDLE handles[] = {startup.StartupInfo.hStdInput, startup.StartupInfo.hStdOutput, err.value};
        SIZE_T bytes = 0;
        InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
        std::vector<unsigned char> attributes(bytes);
        startup.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
        check(InitializeProcThreadAttributeList(startup.lpAttributeList, 1, 0, &bytes));
        struct AttributeCleanup {
            LPPROC_THREAD_ATTRIBUTE_LIST value;
            ~AttributeCleanup() { DeleteProcThreadAttributeList(value); }
        } cleanup{startup.lpAttributeList};
        check(UpdateProcThreadAttribute(startup.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                        handles, sizeof(handles), nullptr, nullptr));
        const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(parent.value), _O_BINARY | (read ? _O_RDONLY : _O_WRONLY));
        if (fd == -1) throw std::runtime_error("cannot open curl pipe");
        parent.value = INVALID_HANDLE_VALUE;  // fd owns it now
        std::unique_ptr<FILE, decltype(&std::fclose)> stream(_fdopen(fd, read ? "rb" : "wb"), &std::fclose);
        if (!stream) { _close(fd); throw std::runtime_error("cannot open curl stream"); }
        // The existing CLI and filesystem::path::string() use the Windows code page.
        const int count = MultiByteToWideChar(CP_ACP, 0, command.c_str(), -1, nullptr, 0);
        check(count != 0);
        std::vector<wchar_t> wide(count);
        check(MultiByteToWideChar(CP_ACP, 0, command.c_str(), -1, wide.data(), count));
        PROCESS_INFORMATION process{};
        check(CreateProcessW(nullptr, wide.data(), nullptr, nullptr, TRUE,
                             CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr,
                             &startup.StartupInfo, &process));
        CloseHandle(process.hThread);
        process_ = process.hProcess;
        file_ = stream.release();
#else
        if (!stdout_file.empty()) command += " > " + quote_curl_arg(stdout_file.string());
        if (!stderr_file.empty()) command += " 2> " + quote_curl_arg(stderr_file.string());
        file_ = popen(command.c_str(), read ? "r" : "w");
        if (!file_) throw std::runtime_error("cannot start curl");
#endif
    }
    ~CurlPipe() { close(); }
    CurlPipe(const CurlPipe&) = delete;
    CurlPipe& operator=(const CurlPipe&) = delete;
    FILE* get() const { return file_; }
    int close() {
        if (!file_) return -1;
#ifdef _WIN32
        std::fclose(file_);
        file_ = nullptr;
        WaitForSingleObject(process_, INFINITE);
        DWORD status = 1;
        GetExitCodeProcess(process_, &status);
        CloseHandle(process_);
        process_ = nullptr;
        return static_cast<int>(status);
#else
        const int status = pclose(file_);
        file_ = nullptr;
        return status != -1 && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
    }
private:
    FILE* file_ = nullptr;
#ifdef _WIN32
    HANDLE process_ = nullptr;
#endif
};

}  // namespace prep
