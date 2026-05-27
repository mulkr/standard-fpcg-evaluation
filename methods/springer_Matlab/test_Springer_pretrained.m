function test_Springer_pretrained(test_dir, output_dir, B_matrix, pi_vector, total_obs_distribution)
    
    springer_options = default_Springer_HSMM_options;

    mkdir(output_dir);
    fprintf("Testing...\n");
    test_files = dir(test_dir + "/*.wav");
    for f = 1:length(test_files)
        file = test_files(f);
        path = string(file.folder)+"/"+string(file.name);
        disp(path);
        [data,fs] = audioread(path);
        data = resample(data,springer_options.audio_Fs,fs);
        states = runSpringerSegmentationAlgorithm(data, springer_options.audio_Fs, B_matrix, pi_vector, total_obs_distribution, false);

        [s1_st,s1_en] = convert_segments(states,1);
        [s2_st,s2_en] = convert_segments(states,3);

         if s1_st(1)>s1_en(1)
            s1_en(1) = [];
        end
        if s1_st(end)>s1_en(end)
            s1_st(end) = [];
        end
        if s2_st(1)>s2_en(1)
            s2_en(1) = [];
        end
        if s2_st(end)>s2_en(end)
            s2_st(end) = [];
        end

        s1_locs = (s1_st+s1_en)/2;
        s2_locs = (s2_st+s2_en)/2;

        s1_locs = s1_locs/springer_options.audio_Fs;
        s2_locs = s2_locs/springer_options.audio_Fs;

        detectfile = fopen(output_dir + "/" + file.name(1:end-4)+".csv","w");
        fprintf(detectfile,"Location;Value\n");
        for s1 = s1_locs
            fprintf(detectfile,string(s1)+";S1\n");
        end
        for s2 = s2_locs
            fprintf(detectfile,string(s2)+";S2\n");
        end
        fclose(detectfile);
    end
    fprintf("Testing: DONE\n");
end
