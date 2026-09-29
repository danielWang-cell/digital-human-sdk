# 1. 构建 Demo App (基础演示)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/demo.cpp")
    add_executable(demo_app demo.cpp)
    # 链接 SDK 核心库
    target_link_libraries(demo_app PRIVATE digital_human_core)
endif()

# 2. 构建 OpenCV 测试程序 (验证图像功能)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/opencv_test.cpp")
    add_executable(opencv_test opencv_test.cpp)
    
    # 链接 SDK 核心库
    # 因为 digital_human_core 在 src 中使用了 PUBLIC 链接 OpenCV，
    # 所以这里会自动获得 OpenCV 的头文件路径和库文件，无需重复 link OpenCV
    target_link_libraries(opencv_test PRIVATE digital_human_core)
endif()

# 3. 构建 ncnn 推理测试
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ncnn_test.cpp")
    add_executable(ncnn_test ncnn_test.cpp)

    # 链接 SDK 核心库
    target_link_libraries(ncnn_test PRIVATE digital_human_core)

endif()

# 4. 构建 FFmpeg 测试程序 (验证音频功能)
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg_audio_test.cpp")
    add_executable(ffmpeg_audio_test ffmpeg_audio_test.cpp)

    # 链接 SDK 核心库
    target_link_libraries(ffmpeg_audio_test PRIVATE digital_human_core)
endif()

# 5. 构建 dlib 测试程序 (验证人脸检测和关键点提取)
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dlib_test.cpp")
    add_executable(dlib_test dlib_test.cpp)

    # 链接 SDK 核心库
    target_link_libraries(dlib_test PRIVATE digital_human_core)
endif()

# 6. ImageLoader Test 
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/image_loader_test.cpp")
    add_executable(image_loader_test image_loader_test.cpp)
    target_link_libraries(image_loader_test PRIVATE digital_human_core)
endif()

# 7. FaceDetector
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/face_detector_test.cpp") 
    add_executable(face_detector_test face_detector_test.cpp)

    target_link_libraries(face_detector_test PRIVATE digital_human_core)
endif()

# 8. landmarks
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/landmark_test.cpp") 
    add_executable(landmark_test landmark_test.cpp)

    target_link_libraries(landmark_test PRIVATE digital_human_core)
endif()

# 9. face_aligner
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/face_aligner_test.cpp") 
    add_executable(face_aligner_test face_aligner_test.cpp)

    target_link_libraries(face_aligner_test PRIVATE digital_human_core)
endif()

# 10. musetalk_engine_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/musetalk_engine_test.cpp") 
    add_executable(musetalk_engine_test musetalk_engine_test.cpp)

    target_link_libraries(musetalk_engine_test PRIVATE digital_human_core)
endif()

# 11. mask_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/mask_test.cpp") 
    add_executable(mask_test mask_test.cpp)

    target_link_libraries(mask_test PRIVATE digital_human_core)
endif()

# 12. musetalk_mask_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/musetalk_mask_test.cpp") 
    add_executable(musetalk_mask_test musetalk_mask_test.cpp)

    target_link_libraries(musetalk_mask_test PRIVATE digital_human_core)
endif()

# 13. musetalk_mask_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/audio_loader_test.cpp") 
    add_executable(audio_loader_test audio_loader_test.cpp)

    target_link_libraries(audio_loader_test PRIVATE digital_human_core)
endif()

# 14. model_validity_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model_validity_test.cpp") 
    add_executable(model_validity_test model_validity_test.cpp)

    target_link_libraries(model_validity_test PRIVATE digital_human_core)
endif()

# 15. audio_framer_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/audio_framer_test.cpp") 
    add_executable(audio_framer_test audio_framer_test.cpp)

    target_link_libraries(audio_framer_test PRIVATE digital_human_core)
endif()

# 16. audio_mel_feature_extract_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/audio_mel_feature_extract_test.cpp") 
    add_executable(audio_mel_feature_extract_test audio_mel_feature_extract_test.cpp)

    target_link_libraries(audio_mel_feature_extract_test PRIVATE digital_human_core)
endif()

# 17. audio_mel_feature_extract_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/audio_preprocess_test.cpp") 
    add_executable(audio_preprocess_test audio_preprocess_test.cpp)

    target_link_libraries(audio_preprocess_test PRIVATE digital_human_core)
endif()

# 18. audio_buffer_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/audio_buffer_test.cpp") 
    add_executable(audio_buffer_test audio_buffer_test.cpp)

    target_link_libraries(audio_buffer_test PRIVATE digital_human_core)
endif()

# 19. model_loader_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model_loader_test.cpp") 
    add_executable(model_loader_test model_loader_test.cpp)

    target_link_libraries(model_loader_test PRIVATE digital_human_core)
endif()

# 20. input_processor_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/input_processor_test.cpp") 
    add_executable(input_processor_test input_processor_test.cpp)

    target_link_libraries(input_processor_test PRIVATE digital_human_core)
endif()

# 21. model_inference_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model_inference_test.cpp")
    add_executable(model_inference_test model_inference_test.cpp)

    target_link_libraries(model_inference_test PRIVATE digital_human_core)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/inference_benchmark.cpp")
    add_executable(inference_benchmark inference_benchmark.cpp)
    target_link_libraries(inference_benchmark PRIVATE digital_human_core)
endif()

# 22. output_processor_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/output_processor_test.cpp") 
    add_executable(output_processor_test output_processor_test.cpp)

    target_link_libraries(output_processor_test PRIVATE digital_human_core)
endif()

# 23. timestamp_manger_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/timestamp_manger_test.cpp") 
    add_executable(timestamp_manger_test timestamp_manger_test.cpp)

    target_link_libraries(timestamp_manger_test PRIVATE digital_human_core)
endif()

# 24. scheduler_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scheduler_test.cpp") 
    add_executable(scheduler_test scheduler_test.cpp)

    target_link_libraries(scheduler_test PRIVATE digital_human_core)
endif()

# 25. audio_video_Synchronous_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/audio_video_Synchronous_test.cpp") 
    add_executable(audio_video_Synchronous_test audio_video_Synchronous_test.cpp)

    target_link_libraries(audio_video_Synchronous_test PRIVATE digital_human_core)
endif()

# 26. pipeline_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pipeline_test.cpp") 
    add_executable(pipeline_test pipeline_test.cpp)

    target_link_libraries(pipeline_test PRIVATE digital_human_core)
endif()

# 27. audio_process_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/audio_process_test.cpp") 
    add_executable(audio_process_test audio_process_test.cpp)

    target_link_libraries(audio_process_test PRIVATE digital_human_core)
endif()

# 28. inference_process_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/inference_process_test.cpp") 
    add_executable(inference_process_test inference_process_test.cpp)

    target_link_libraries(inference_process_test PRIVATE digital_human_core)
endif()

# 29. render_process_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/render_process_test.cpp") 
    add_executable(render_process_test render_process_test.cpp)

    target_link_libraries(render_process_test PRIVATE digital_human_core)
endif()

# 30. ffmpeg_pulse_capture_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg_pulse_capture_test.cpp") 
    add_executable(ffmpeg_pulse_capture_test ffmpeg_pulse_capture_test.cpp)

    target_link_libraries(ffmpeg_pulse_capture_test PRIVATE digital_human_core)
endif()

# 31. realtime_audio_processor_test
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/realtime_audio_processor_test.cpp") 
    add_executable(realtime_audio_processor_test realtime_audio_processor_test.cpp)

    target_link_libraries(realtime_audio_processor_test PRIVATE digital_human_core)
endif()


message(STATUS "Configured example targets.")
