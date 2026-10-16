// Windows part of every model launcher run.exe (src/models/<model>/host.cpp calls <model>_host_main in
// the prebuilt host_<model>.dll). build_windows.bat delay-loads that DLL, and this hook loads it from
// ..\..\lib\ beside the launcher's folder, as the Ubuntu run.exe finds libhost_<model>.so through its
// RUNPATH $ORIGIN/../../lib: src/exe_models_win/<model>/run.exe -> src/lib in the repo,
// server_running/exe_models/<model>/run.exe -> server_running/lib in a package. The DLLs the host DLL
// imports (Boost, XRT, LLVM OpenMP, the VC++ runtime) are found in the same folder.
#include <windows.h>

#include <delayimp.h>

#include <cstdio>
#include <string>

namespace {

std::wstring lib_dir() {
    std::wstring exe(32768, L'\0');
    exe.resize(GetModuleFileNameW(nullptr, exe.data(), static_cast<DWORD>(exe.size())));
    const std::wstring dir = exe.substr(0, exe.find_last_of(L"\\/") + 1) + L"..\\..\\lib\\";
    std::wstring full(32768, L'\0');
    full.resize(GetFullPathNameW(dir.c_str(), static_cast<DWORD>(full.size()), full.data(), nullptr));
    return full;
}

FARPROC WINAPI on_delay_load(unsigned notify, PDelayLoadInfo info) {
    switch (notify) {
    case dliNotePreLoadLibrary: {
        std::wstring path = lib_dir();
        for (const char* c = info->szDll; *c; ++c) path += static_cast<wchar_t>(static_cast<unsigned char>(*c));
        // Null when it is not there: the default search (run.exe's folder, then PATH) is tried next.
        return reinterpret_cast<FARPROC>(
            LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS));
    }
    case dliFailLoadLib:
        std::fprintf(stderr, "run.exe: cannot load %s from %ls or the default DLL search path (Windows error %lu)\n",
                     info->szDll, lib_dir().c_str(), info->dwLastError);
        break;
    case dliFailGetProc:
        std::fprintf(stderr, "run.exe: %s has no entry point %s; the launcher and host DLL do not match\n",
                     info->szDll, info->dlp.fImportByName ? info->dlp.szProcName : "(by ordinal)");
        break;
    default:
        return nullptr;
    }
    std::fflush(stderr);
    ExitProcess(127);
}

}  // namespace

extern "C" const PfnDliHook __pfnDliNotifyHook2 = on_delay_load;
extern "C" const PfnDliHook __pfnDliFailureHook2 = on_delay_load;
