set(TARGET ttft_benchmark)
add_executable(${TARGET} ttft_benchmark.cpp)

# 链接必要的库
target_link_libraries(${TARGET} PRIVATE 
    common 
    llama
)

# 包含头文件目录
target_include_directories(${TARGET} PRIVATE 
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/include
)

# 设置编译特性
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# 如果编译为 Release 模式，启用优化
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(${TARGET} PRIVATE -O3)
endif()

# 安装规则（可选）
install(TARGETS ${TARGET} RUNTIME DESTINATION bin)
