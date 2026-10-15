// Windows stand-in for a worker in test_server.py, which cannot start a Python script named
// run.exe: runs "<VGF_TEST_PYTHON> <this folder>\run.py <the same arguments>" with the same
// working directory and standard handles, and returns its exit code. Not used by vgf-serve.
#include <windows.h>

#include <string>
#include <vector>

int wmain() {
    // Keep the arguments exactly as received: skip only the program name, the way the
    // C runtime does (a quoted token, or everything up to the first space or tab).
    std::wstring arguments = GetCommandLineW();
    std::size_t start = 0;
    if (!arguments.empty() && arguments[0] == L'"') {
        const std::size_t close = arguments.find(L'"', 1);
        start = close == std::wstring::npos ? arguments.size() : close + 1;
    } else {
        while (start < arguments.size() && arguments[start] != L' ' && arguments[start] != L'\t') ++start;
    }
    arguments.erase(0, start);

    std::vector<wchar_t> module(32768);
    const DWORD length = GetModuleFileNameW(nullptr, module.data(), static_cast<DWORD>(module.size()));
    std::wstring script(module.data(), length);
    script = script.substr(0, script.find_last_of(L"\\/") + 1) + L"run.py";

    std::vector<wchar_t> python(32768);
    std::wstring interpreter = L"python";
    if (const DWORD size = GetEnvironmentVariableW(L"VGF_TEST_PYTHON", python.data(),
                                                   static_cast<DWORD>(python.size()));
        size > 0 && size < python.size()) {
        interpreter.assign(python.data(), size);
    }

    std::wstring command = L"\"" + interpreter + L"\" \"" + script + L"\"" + arguments;
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr,
                        &startup, &process)) {
        return 127;
    }
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(process.hProcess, &exit_code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return static_cast<int>(exit_code);
}
