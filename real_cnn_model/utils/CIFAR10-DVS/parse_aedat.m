function []=parse_aedat(file_name)
    new_root = 'mat_data';
    old_root = 'data'; 
    file_list = textscan(fopen(file_name), '%s');
    file_list = file_list{1};
    for i=1:length(file_list)
        event_data = dat2mat(file_list{i});
        event_data = event_data(:, [1, 4, 5, 6]);
        save_name = replace(file_list{i}, old_root, new_root);
        save_name = replace(save_name, 'aedat', 'mat');
        save(save_name, 'event_data')
    end
    

end
