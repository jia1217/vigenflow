### Option 2: Server Mode (API-Based)

This method compiles and runs an HTTP server that listens for generation requests. When a request is received via POST, the server handles the execution of the underlying NPU worker and returns the generated image data.


#### Step 1: Compile the Server

Use `g++` to compile the server code. Ensure you link the necessary threads and Boost libraries.

```bash
g++ -std=c++17 -O2 main.cpp -lpthread -lboost_system -o server
```

#### Step 3: Start the Server

Run the generated server binary. You must specify the base paths for the weights and NPU files using command-line flags. These paths act as the global defaults for the server.

```bash
./server -h
```

By default, the server will run and listen on `http://0.0.0.0:1234`.