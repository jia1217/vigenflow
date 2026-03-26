##############################################add addition link package###########################################
CXXFLAGS += -I/usr/local/include -I/home/kelsey/NPU_projects/NPU_new/new_test/AUser_host/external_libs/tokenizers-cpp/include
# Add the path to where the compiled library (.a or .so file) lives
LDFLAGS += -L/home/kelsey/NPU_projects/NPU_new/new_test/AUser_host/external_libs/tokenizers-cpp/build

# Tell the linker the name of the library to link
LDLIBS += -ltokenizers_cpp
##################################################################################################################