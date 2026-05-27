function [B_matrix, pi_vector] = train_test_Schmidt(input_dir, test_dir, output_dir)

    schmidt_options = default_Schmidt_HSMM_options;
    files = dir(input_dir + "/*.wav");
    labs = {};
    dats = {};

    fprintf("Loading data...\n");
    for f = 1:length(files)
        file = files(f);
        path = string(file.folder)+"/"+string(file.name);
        disp(path);
        labfile = input_dir+"/"+file.name(1:end-4)+".csv";
        labtable = readtable(labfile,"Delimiter",";");
        [data,fs] = audioread(path);
        data = resample(data,schmidt_options.audio_Fs,fs);
        S1 = round(labtable.Location(labtable.Value == "S1")*schmidt_options.audio_segmentation_Fs);
        S2 = round(labtable.Location(labtable.Value == "S2")*schmidt_options.audio_segmentation_Fs);
        labs{f,1} = S1;
        labs{f,2} = S2;
        dats{f} = data;
    end
    fprintf("Loading data: DONE\n");

    fprintf("Training...\n");
    [B_matrix, pi_vector] = ...
        trainSchmidtSegmentationAlgorithm(dats,labs,schmidt_options.audio_Fs, false); %?
    fprintf("Training: DONE\n");

    fprintf("Testing...\n");
    test_files = dir(test_dir + "/*.wav");
    for f = 1:length(test_files)
        file = test_files(f);
        path = string(file.folder)+"/"+string(file.name);
        disp(path);
        [data,fs] = audioread(path);
        states = runSchmidtSegmentationAlgorithm(data, schmidt_options.audio_Fs, B_matrix, pi_vector, false);

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

        s1_locs = s1_locs/fs;
        s2_locs = s2_locs/fs;

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
