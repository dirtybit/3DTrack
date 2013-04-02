INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/libaruco ${GNULIBS_INCLUDE_DIR})
LINK_LIBRARIES(${PROJECT_NAME} ${REQUIRED_LIBRARIES} )

ADD_EXECUTABLE(tracker tracker.cpp)
ADD_EXECUTABLE(aruco_debug aruco_debug.cpp)
ADD_EXECUTABLE(aruco_create_marker aruco_create_marker.cpp)

INSTALL(TARGETS tracker aruco_debug aruco_create_marker RUNTIME DESTINATION bin)


