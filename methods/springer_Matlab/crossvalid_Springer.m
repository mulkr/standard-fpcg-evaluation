function [B_mats, pi_vecs, tot_obs] = crossvalid_Springer(input_dir, output_dir)
    FOLDS = 10;

    springer_options = default_Springer_HSMM_options;
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
        % data = resample(data,schmidt_options.audio_Fs,fs);
        S1 = round(labtable.Location(labtable.Value == "S1")*springer_options.audio_segmentation_Fs);
        S2 = round(labtable.Location(labtable.Value == "S2")*springer_options.audio_segmentation_Fs);
        labs{f,1} = S1;
        labs{f,2} = S2;
        dats{f} = data;
    end
    fprintf("Loading data: DONE\n");

    B_mats = [];
    pi_vecs = [];
    tot_obs = [];
    fprintf("Crossvalidation...\n");
    for k = 1:FOLDS
        fprintf("Training fold %d...\n",k);
        fold_size = length(files)/FOLDS;
        test_ind = (1:fold_size)+fold_size*(k-1);
        train_ind = 1:length(files);
        train_ind(test_ind) = [];
        [B_matrix, pi_vector, total_obs_distribution] = ...
            trainSpringerSegmentationAlgorithm(dats(train_ind),labs(train_ind,:),springer_options.audio_Fs, false);
        B_mats{k} = B_matrix;
        pi_vecs{k} = pi_vector;
        tot_obs{k} = total_obs_distribution;
        fprintf("Training fold %d: DONE\n",k);

        fprintf("Testing fold %d...\n",k);
        for file = files(test_ind)'
            path = string(file.folder)+"/"+string(file.name);
            disp(path);
            [data,fs] = audioread(path);
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

            s1_locs = s1_locs/fs;
            s2_locs = s2_locs/fs;

            detectfile = fopen(output_dir +"/" + file.name(1:end-4)+".csv","w");
            fprintf(detectfile,"Location;Value\n");
            for s1 = s1_locs
                fprintf(detectfile,string(s1)+";S1\n");
            end
            for s2 = s2_locs
                fprintf(detectfile,string(s2)+";S2\n");
            end
            fclose(detectfile);
        end
        fprintf("Testing fold %d: DONE\n",k);
    end
    fprintf("Crossvalidation: DONE\n");
end
