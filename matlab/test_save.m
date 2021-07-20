function [] = test_save(nbit, dset, prefix)
    n_bits = nbit;
    dataset = dset;
    addpath('utils/');
    fprintf('Load hashcodes and labels of %s.\n', dataset);
    hashcode_path = sprintf('../ARGA/hashcode/GCNH_%s_%dbits.mat', dataset, n_bits);

    load (hashcode_path);
    % label
    if strcmp(dataset, 'COCO')
        load('/media/00_database/coco/vgg/retrieval2017.mat', 'retrieval_label');
        load('/media/00_database/coco/vgg/val2017.mat', 'val_label');
    elseif strcmp(dataset, 'NUSWIDE')
        load('/media/00_database/nuswide/vgg/retrieval.mat', 'retrieval_label');
        load('/media/00_database/nuswide/vgg/val.mat', 'val_label');
    end

    trn_label = double(retrieval_label);
    tst_label = double(val_label);
    cateRetriTest = sign(double(trn_label) * double(tst_label')) == 1;
   
    fid = fopen('../ARGA/log_GAEH_test_time.txt', 'at');
    tmp_T = tic;             % time start 
    B = compactbit(retrieval_B > 0);
    tB = compactbit(val_B > 0);
    hamm = hammingDist(tB, B)';
    [~, hammRetriTest] = sort(hamm, 1);
    [precision, recall, map] = fast_PR_MAP(int32(cateRetriTest), int32(hammRetriTest));
    test_time = toc(tmp_T); % time end

    in_file = ['---------', dataset, '----', num2str(n_bits),'----start------'];
    fprintf(fid,'%s', in_file);
    fprintf(fid,'\r\n'); 
    in_file = ['test time: ', num2str(test_time)];
    fprintf(fid,'%s', in_file);
    fprintf(fid,'\r\n'); 
    fclose(fid);   
    
    clear hamm;
    clear cateRetriTest;
    
   % for draw 
    draw_pre_topk = zeros(20);
    draw_rec_topk = zeros(20);
    draw_map_topk = zeros(20);
    for i=50:50:1000
        draw_pre_topk(floor(i/50)) = precision(i);
        draw_rec_topk(floor(i/50)) = recall(i);
        draw_map_topk(floor(i/50)) = map(i);
    end
    
    map_mean = map(1, size(hammRetriTest, 1));
    clear hammRetriTest;
    
    % topk results
    topk = 1000;
    map_topk       = map(topk); 
    precision_topk = precision(topk); 
    recall_topk    = recall(topk);         

    % the final results
    result.map_mean       = map_mean;
    result.(['map_', num2str(topk)])      = map_topk;
    result.(['precision_', num2str(topk)]) = precision_topk;
    result.(['recall_', num2str(topk)])    = recall_topk;

    % save
    result_name = ['../results/' prefix '_' dataset '_' num2str(n_bits) 'bits_' datestr(now,30) '.mat'];
    save(result_name, 'precision', 'recall', 'map', 'result', ...
        'draw_pre_topk', 'draw_rec_topk', 'draw_map_topk');
end






