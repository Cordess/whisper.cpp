if (WHISPER_SDL2)
    # command
    set(CMAKE_CXX_STANDARD 17)
    set(DEFAULT_BUILD_TYPE "Release")
    if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
      set(DEFAULT_BUILD_TYPE "Debug")
    endif()
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
      message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
      set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
      # Set the possible values of build type for cmake-gui
      set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
    endif()

    #set(TARGET command)
    set(TARGET whisper-command)
    add_executable(${TARGET} command.cpp) #wenn executable gebraucht wird
    #add_library(${TARGET} SHARED command.cpp)#dll variante
    if(WIN32) # Install dlls in the same directory as the executable on Windows
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    endif()

    include(DefaultTargetOptions)
    include(FetchContent)
    FetchContent_Declare(cpr GIT_REPOSITORY https://github.com/libcpr/cpr.git
                         GIT_TAG 1.12.0) # Replace with your desired git commit from: https://github.com/libcpr/cpr/releases
    FetchContent_MakeAvailable(cpr)
    include(/Users/Cordess/source/repos/VCPKG/vcpkg/scripts/buildsystems/vcpkg.cmake)
    link_directories(/Users/Cordess/source/repos/VCPKG/vcpkg/installed/x64-windows/lib)
    #find_package(cpr CONFIG REQUIRED)
    target_link_libraries(${TARGET} PRIVATE common common-sdl whisper ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(${TARGET} PRIVATE cpr::cpr)

    install(TARGETS ${TARGET} RUNTIME)
endif ()
