if(APPLE)
    # Initialize CMake flags if not already set
    if(NOT CMAKE_CXX_FLAGS)
        set(CMAKE_CXX_FLAGS "")
    endif()
    
    # Use libc++ as the standard library
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    
    # Get the SDK path
    execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE SDKROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    # Set the sysroot
    set(CMAKE_OSX_SYSROOT "${SDKROOT}")
    
    # Add the correct include paths for C++ standard library headers
    include_directories(SYSTEM
        "${SDKROOT}/usr/include/c++"
        "${SDKROOT}/usr/include"
    )
endif() 