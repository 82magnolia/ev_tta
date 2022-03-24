function []=parse_dat(file_name)
    new_root = 'mat_data';
    old_root = 'data'; 
    file_list = textscan(fopen(file_name), '%s');
    file_list = file_list{1};
    for i=1:length(file_list)
        event_data = load_atis_data(file_list{i});
        event_data = [event_data.x, event_data.y, event_data.ts, event_data.p];
        save_name = replace(file_list{i}, old_root, new_root);
        save_name = replace(save_name, '.dat', '.mat');
        save(save_name, 'event_data')
    end
end
