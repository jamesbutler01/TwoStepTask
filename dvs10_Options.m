%% Discrete value space task
% EG 2022

hotkey('x', 'escape_screen(); assignin(''caller'',''continue_'',false);');

% detect an available tracker
if exist('eye_','var'), tracker = eye_;
elseif exist('eye2_','var'), tracker = eye2_;
else, error('This demo requires eye input. Please set it up or turn on the simulation mode.');
end

%% Trial parameters
% time intervals (in ms):
max_for_fix = 5000;             % max allowed for initial fixation
initial_fix = 500;
options_on_time = 800;          % how long each option is on for
max_choice_rt = 9000;           % max allowed for choice
hold_target_time = 500;
nofix_timeout = 3000;           % timeout for no fixation
nochoice_timeout = 6000;        % timeout for rejected choice
iti = 1000;                     

% fixation window (in degrees):
fix_radius = 3.5;      % fixation point
hold_opt   = 3.5;      % targets

% background colours
white  = [1 1 1];
brown  = [0.4 0.2 0];
yellow = [1 1 0];
purple = [0.698 0.4 1];
bg_trial   = white;      % standard trial background
bg_nofix   = brown;      % missed fixation 
bg_correct = yellow;     % correct choice
bg_wrong   = purple;     % wrong choice

% option positions on the screen
cues_x_positions = [-10 10];
cues_y_positions = [6 -6];

% parameters for boxes surrounding options
box_size = [5 16];

% editables
stimulus_set = 2;
magnitude_reward_increments = 250;      % ms
probability_reward_ms = 1200;
editable('stimulus_set')
editable('magnitude_reward_increments')
editable('probability_reward_ms')

%% Space parameters
% space structure
dimension = 10;             % size of each dimension
nOptions  = dimension^2;    % number of options
nSets     = 2;              % number of stimulus sets
optionIDs = reshape(1:nOptions,dimension,dimension);

%% Event codes
% 0:256, 9 and 18 reserved

% outcome codes - must be between 0 and 9
ec_out_goodtrial = 0;
ec_out_nofix = 4;               % trial not started (no initial fixation)
ec_out_nochoice = 1;
ec_out_correctchoice = 5;
ec_out_incorrectchoice = 6;

% trial event codes
ec_fix_screen_shown = 10;
ec_timeout_screenshown = 11;
ec_fbscreenshown = 12;
ec_iti_start = 16;
ec_iti_end = 17;

ec_option1_on  = 20;
ec_option1made = 21;

ec_option2_on  = 22;
ec_option2made = 23;

ec_fix2_on  = 24;
ec_fix2made = 25;

ec_choice_on  = 26;
ec_choicemade = 27;

ec_mag1_fix = 28;
ec_prob1_fix = 29;
ec_mag2_fix = 38;
ec_prob2_fix = 39;

ec_upleft_fix = 48;
ec_downleft_fix = 49;
ec_upright_fix = 58;
ec_downright_fix = 59;

ec_unrewarded = 30;
ec_pump_on = 7;
ec_pump_off = 8;

% option id codes
ec_option1 = 40;                   % the next ec will be from ec_optionids and will relate to this
ec_option2 = 41;                   
ec_optionids = reshape(256-nOptions+1:256,dimension,dimension);

ec_option1_loc = 42;
ec_option2_loc = 43;
ec_left_right_locs  = [52,53];
ec_mag1_loc = 44;
ec_mag2_loc = 45;
ec_up_down_locs = [54,55];

%% Initialise trial variables
% rts
dvs_rts_mag1 = [];
dvs_ts_mag1_acquired = [];
dvs_mag1_held = [];
dvs_rts_prob1 = [];
dvs_ts_prob1_acquired = [];
dvs_prob1_held = [];

dvs_rts_mag2 = [];
dvs_ts_mag2_acquired = [];
dvs_mag2_held = [];
dvs_rts_prob2 = [];
dvs_ts_prob2_acquired = [];
dvs_prob2_held = [];

dvs_rts_choice = []; 
dvs_ts_choice_acquired = [];
dvs_rts_mag3 = [];
dvs_ts_mag3_acquired = [];
dvs_mag3_held = [];
dvs_rts_prob3 = [];
dvs_ts_prob3_acquired = [];
dvs_prob3_held = [];
dvs_rts_mag4 = [];
dvs_ts_mag4_acquired = [];
dvs_mag4_held = [];
dvs_rts_prob4 = [];
dvs_ts_prob4_acquired = [];
dvs_prob4_held = [];

% options
dvs_left_value  = [];
dvs_left_mag    = [];
dvs_left_prob   = [];
dvs_right_value = [];
dvs_right_mag   = [];
dvs_right_prob  = [];

% choice
dvs_chosen_value   = [];
dvs_chosen_mag     = [];
dvs_chosen_prob    = [];
dvs_unchosen_value = [];
dvs_unchosen_mag   = [];
dvs_unchosen_prob  = [];
dvs_chose_left     = [];
dvs_chose_option1  = [];

if TrialRecord.CurrentTrialNumber == 1
    TrialRecord.User.num_initiated_trials = 0;      % first fixation acquired
    TrialRecord.User.num_choicemade_trials = 0;
    TrialRecord.User.num_correct_trials = 0;        % higher value target chosen
    TrialRecord.User.perc_correct_trials = 0;

    TrialRecord.User.block_trial_counter = 1; 
end

%% Predetermine choices for the session
if TrialRecord.CurrentTrialNumber == 1
%     choices_set = readmatrix('choices_set.txt');
%     choices_made = sum(readmatrix('choices_made.txt'));
% 
%     if choices_made==0                                                      % start from the first if it's the first session
%         TrialRecord.User.trial_sequence = choices_set(1:5000,:);                            % make available more choices than will ever be in a session
%     else
%         TrialRecord.User.trial_sequence = choices_set(choices_made+1:choices_made+5000,:);  % pick up from where we left off last session
%     end

TrialRecord.User.option1_available = 1:100;     % set up all possible options to draw from
TrialRecord.User.option2_available = 1:100;

end
%% Select trial cues and positions 
% subselect task objects based on which options they'll be used for (ML doesn't allow the same task object to be shown twice on screen, 
% so each cue is saved twice)
if stimulus_set==1
    add_this = 0;
elseif stimulus_set==2
    add_this = dimension*4;
else
    ME = MException('myComponent:TaskError', 'Non-existent "stimulus_set" parameter. Select 1 for stimulus set 1, or 2 for stimulus set 2.');
    throw(ME);
end
option1_available_task_objects = [1+add_this:dimension+add_this; dimension+1+add_this:dimension*2+add_this];                    % [magnitude cues;probability cues]
option2_available_task_objects = [dimension*2+add_this+1:dimension*3+add_this; dimension*3+add_this+1:dimension*4+add_this];

% select the options for the trial 
opt1_i = 999;
opt2_i = 999;

if length(TrialRecord.User.option1_available)==2                                                        % if there's only two options available left
    are_any_same = ismember(TrialRecord.User.option1_available,TrialRecord.User.option2_available);     % check if they're the same
    if sum(are_any_same)==1
        opt1_i = TrialRecord.User.option1_available(are_any_same);                                      % if there's one option that's the same, pick the other one
        opt2_i = TrialRecord.User.option2_available(TrialRecord.User.option2_available~=opt1_i);
    elseif sum(are_any_same)==2
        pick_opt = randperm(2);                                                                         % if they're both the same, pick the opposite one
        opt1_i = TrialRecord.User.option1_available(pick_opt(1));
        opt2_i = TrialRecord.User.option2_available(pick_opt(2));
    else
        while opt1_i==opt2_i
        opt1_i = TrialRecord.User.option1_available(randi(length(TrialRecord.User.option1_available)));
        opt2_i = TrialRecord.User.option2_available(randi(length(TrialRecord.User.option2_available)));
        end
    end
else
    while opt1_i==opt2_i
        opt1_i = TrialRecord.User.option1_available(randi(length(TrialRecord.User.option1_available)));
        opt2_i = TrialRecord.User.option2_available(randi(length(TrialRecord.User.option2_available)));
    end
end

TrialRecord.User.option1_available(TrialRecord.User.option1_available==opt1_i) = [];                      % once picked, eliminate it from the list of available options
TrialRecord.User.option2_available(TrialRecord.User.option2_available==opt2_i) = [];

if isempty(TrialRecord.User.option1_available)                                           % check if all options have been picked
    TrialRecord.User.option1_available = 1:100;
    TrialRecord.User.option2_available = 1:100;
end

[mag1_i,prob1_i] = find(optionIDs==opt1_i);             % from the option ID code (1-100) extract magnitude and probability
[mag2_i,prob2_i] = find(optionIDs==opt2_i);

mag1  = option1_available_task_objects(1,mag1_i);       % pick the magnitude cue out of the ones available for this option
prob1 = option1_available_task_objects(2,prob1_i);
mag2  = option2_available_task_objects(1,mag2_i);
prob2 = option2_available_task_objects(2,prob2_i);

[option1_value] = mag1_i*prob1_i;
[option2_value] = mag2_i*prob2_i;

log_option_ids(ec_option1,mag1_i,prob1_i)
log_option_ids(ec_option2,mag2_i,prob2_i)

% select option left and right position (x)
x_position1_i = 999; 
x_position2_i = 999;
while x_position1_i == x_position2_i
    x_position1_i = randi(length(cues_x_positions));
    x_position2_i = randi(length(cues_x_positions));
end
option1_x_position = cues_x_positions(x_position1_i);
option2_x_position = cues_x_positions(x_position2_i);

log_cue_locs(ec_option1_loc,ec_left_right_locs,x_position1_i)      
log_cue_locs(ec_option2_loc,ec_left_right_locs,x_position2_i)

% select cue up and down position for magnitude/probability (y)
mag1_y_position_i  = 999; 
prob1_y_position_i = 999;
while mag1_y_position_i == prob1_y_position_i
    mag1_y_position_i  = randi(length(cues_y_positions));
    prob1_y_position_i = randi(length(cues_y_positions));
end
mag1_y_position = cues_y_positions(mag1_y_position_i);
prob1_y_position = cues_y_positions(prob1_y_position_i);

% if we wanted to randomise up/down for mag/prob between option 1 and 2
% mag2_y_position_i  = 999;                                     
% prob2_y_position_i = 999;
% while mag2_y_position_i == prob2_y_position_i
%     mag2_y_position_i  = randi(length(cues_y_positions));
%     prob2_y_position_i = randi(length(cues_y_positions));
% end
mag2_y_position = cues_y_positions(mag1_y_position_i);
prob2_y_position = cues_y_positions(prob1_y_position_i);

log_cue_locs(ec_mag1_loc,ec_up_down_locs,mag1_y_position_i)      
log_cue_locs(ec_mag2_loc,ec_up_down_locs,mag1_y_position_i)

% other task objects
fixation_point = dimension*8+1;   
option_fix1    = dimension*8+2;   
option_fix2    = dimension*8+3; 

%% Create scenes
% scene 1: fixation
fix = SingleTarget(tracker);   
fix.Target = fixation_point;  
fix.Threshold = fix_radius;   
fixth = FreeThenHold(fix);     
fixth.MaxTime  = max_for_fix;   
fixth.HoldTime = initial_fix;                               

fixation_scene = create_scene(fixth,fixation_point); 

% scene 2: option 1 on
reposition_object(mag1,[option1_x_position mag1_y_position])
reposition_object(prob1,[option1_x_position prob1_y_position])

box1 = BoxGraphic(null_);                   % surround option with a box
box1.EdgeColor = [0 0 0];
box1.FaceColor = [1 1 1];
box1.Size = box_size;
box1.Position = [option1_x_position 0];

mag_fix1 = SingleTarget(tracker);           % keep track of what they look at first
mag_fix1.Target = mag1;
mag_fix1.Threshold = hold_opt;
mag_fth1 = FreeThenHold(mag_fix1);
mag_fth1.MaxTime = options_on_time;
mag_fth1.HoldTime = 1;
fta_mag1 = FixTimeAnalyzer(mag_fth1);       % how long did they fixate it for?

prob_fix1 = SingleTarget(tracker);
prob_fix1.Target = prob1;
prob_fix1.Threshold = hold_opt;
prob_fth1 = FreeThenHold(prob_fix1);
prob_fth1.MaxTime = options_on_time;
prob_fth1.HoldTime = 1;
fta_prob1 = FixTimeAnalyzer(prob_fth1);

tc1 = TimeCounter(tracker);                  % regardless of fixation behaviour, scene continues for this duration 
tc1.Duration = options_on_time;

con_opt1 = Concurrent(fta_mag1);            % add box around option 
con_opt1.add(box1)

any_cont = AnyContinue(tc1);                % scene continues if any adapter continues (i.e. until tc expires)
any_cont.add(fta_prob1)
any_cont.add(con_opt1)

option1_scene = create_scene(any_cont,[mag1 prob1]);

% scene 3: cue 2 on
reposition_object(mag2,[option2_x_position mag2_y_position])
reposition_object(prob2,[option2_x_position prob2_y_position])

box2 = BoxGraphic(null_);                   % surround option with a box
box2.EdgeColor = [0 0 0];
box2.FaceColor = [1 1 1];
box2.Size = box_size;
box2.Position = [option2_x_position 0];

mag_fix2 = SingleTarget(tracker);           % keep track of what they look at first
mag_fix2.Target = mag2;
mag_fix2.Threshold = hold_opt;
mag_fth2 = FreeThenHold(mag_fix2);
mag_fth2.MaxTime = options_on_time;
mag_fth2.HoldTime = 1;
fta_mag2 = FixTimeAnalyzer(mag_fth2);

prob_fix2 = SingleTarget(tracker);
prob_fix2.Target = prob2;
prob_fix2.Threshold = hold_opt;
prob_fth2 = FreeThenHold(prob_fix2);
prob_fth2.MaxTime = options_on_time;
prob_fth2.HoldTime = 1;
fta_prob2 = FixTimeAnalyzer(prob_fth2);


tc2 = TimeCounter(tracker);                  % regardless of behaviour, scene continues for this duration 
tc2.Duration = options_on_time;

con_opt2 = Concurrent(fta_mag2);            % add box around option 
con_opt2.add(box2)

any2_cont = AnyContinue(tc2);                 % scene continues if any adapter continues
any2_cont.add(fta_prob2)
any2_cont.add(con_opt2)

option2_scene = create_scene(any2_cont,[mag2 prob2]);

% scene 4: reacquire central fixation
fix2 = SingleTarget(tracker);  
fix2.Target = fixation_point;  
fix2.Threshold = fix_radius;   

fixth2 = FreeThenHold(fix);     
fixth2.MaxTime  = max_choice_rt;   
fixth2.HoldTime = hold_target_time;                               

fixation2_scene = create_scene(fixth2,fixation_point);

% scene 5: choice 
reposition_object(mag1,[option1_x_position mag1_y_position])
reposition_object(prob1,[option1_x_position prob1_y_position])

reposition_object(mag2,[option2_x_position mag2_y_position])
reposition_object(prob2,[option2_x_position prob2_y_position])

reposition_object(option_fix1,[option1_x_position 0])
reposition_object(option_fix2,[option2_x_position 0])

% for each option: box, big target for choice, inactive targets to check individual cue fixation
box3 = BoxGraphic(null_);
box3.EdgeColor = [0 0 0];
box3.FaceColor = [1 1 1];
box3.Size = box_size;
box3.Position = [option1_x_position 0];

f3 = SingleTarget(tracker);        
f3.Target = box3;             
f3.Threshold = box_size;
fth3 = FreeThenHold(f3);
fth3.MaxTime = max_choice_rt;
fth3.HoldTime = hold_target_time;

mag_fix3 = SingleTarget(tracker);           
mag_fix3.Target = mag1;
mag_fix3.Threshold = hold_opt;
mag_fth3 = FreeThenHold(mag_fix3);
mag_fth3.MaxTime = options_on_time;
mag_fth3.HoldTime = 1;
fta_mag3 = FixTimeAnalyzer(mag_fth3);

prob_fix3 = SingleTarget(tracker);
prob_fix3.Target = prob1;
prob_fix3.Threshold = hold_opt;
prob_fth3 = FreeThenHold(prob_fix3);
prob_fth3.MaxTime = options_on_time;
prob_fth3.HoldTime = 1;
fta_prob3 = FixTimeAnalyzer(prob_fth3);

con_opt1_ch = Concurrent(fth3);
con_opt1_ch.add(box3)
con_opt1_ch.add(fta_mag3)
con_opt1_ch.add(fta_prob3)

% option 2
box4 = BoxGraphic(null_);
box4.EdgeColor = [0 0 0];
box4.FaceColor = [1 1 1];
box4.Size = box_size;
box4.Position = [option2_x_position 0];

f4 = SingleTarget(tracker);        
f4.Target = box4;             
f4.Threshold = box_size;
fth4 = FreeThenHold(f4);
fth4.MaxTime = max_choice_rt;
fth4.HoldTime = hold_target_time;

mag_fix4 = SingleTarget(tracker);           
mag_fix4.Target = mag2;
mag_fix4.Threshold = hold_opt;
mag_fth4 = FreeThenHold(mag_fix4);
mag_fth4.MaxTime = options_on_time;
mag_fth4.HoldTime = 1;
fta_mag4 = FixTimeAnalyzer(mag_fth4);

prob_fix4 = SingleTarget(tracker);
prob_fix4.Target = prob2;
prob_fix4.Threshold = hold_opt;
prob_fth4 = FreeThenHold(prob_fix4);
prob_fth4.MaxTime = options_on_time;
prob_fth4.HoldTime = 1;
fta_prob4 = FixTimeAnalyzer(prob_fth4);

con_opt2_ch = Concurrent(fth4);
con_opt2_ch.add(box4)
con_opt2_ch.add(fta_mag4)
con_opt2_ch.add(fta_prob4)

or = OrAdapter(con_opt1_ch);            % scene moves on if one OR the other succeed
or.add(con_opt2_ch);

tc_ch = TimeCounter(tracker);           % scene times out if neither succeeds 
tc_ch.Duration = max_choice_rt;
or.add(tc_ch);

choice_scene = create_scene(or,[mag1 prob1 mag2 prob2 option_fix1 option_fix2]);

% scene 5: feedback 
% set it up but don't create it since cue shown depends on option chosen
box5 = BoxGraphic(null_);
box5.EdgeColor = [0 0 0];
box5.FaceColor = [0 0 0];
box5.Size = box_size;

f5 = SingleTarget(tracker);        
f5.Target = [999 999];                  % can't be chosen             
f5.Threshold = hold_opt;

%% TASK
set_bgcolor(bg_trial);      % set trial background
dashboard(1, sprintf('Number trials initiated: %0.0f',TrialRecord.User.num_initiated_trials), white);
dashboard(2, sprintf('Number trials attempted: %0.0f',TrialRecord.User.num_choicemade_trials), white);
dashboard(3, sprintf('Performance: %0.0f%% correct',TrialRecord.User.perc_correct_trials), white)

% save trial variables 
if x_position1_i==1                                     % option 1 is on the left
    dvs_left_value  = [dvs_left_value option1_value];
    dvs_left_mag    = [dvs_left_mag mag1_i];
    dvs_left_prob   = [dvs_left_prob prob1_i];
    dvs_right_value = [dvs_right_value option2_value];
    dvs_right_mag   = [dvs_right_mag mag2_i];
    dvs_right_prob  = [dvs_right_prob prob2_i];
elseif x_position1_i==2                                 % option 1 is on the right
    dvs_left_value  = [dvs_left_value option2_value];
    dvs_left_mag    = [dvs_left_mag mag2_i];
    dvs_left_prob   = [dvs_left_prob prob2_i];
    dvs_right_value = [dvs_right_value option1_value];
    dvs_right_mag   = [dvs_right_mag mag1_i];
    dvs_right_prob  = [dvs_right_prob prob1_i];
end
bhv_variable('dvs_left_value', dvs_left_value);
bhv_variable('dvs_left_mag', dvs_left_mag);
bhv_variable('dvs_left_prob', dvs_left_prob);
bhv_variable('dvs_right_value', dvs_right_value);
bhv_variable('dvs_right_mag', dvs_right_mag);
bhv_variable('dvs_right_prob', dvs_right_prob);
bhv_variable('stimulus_set',stimulus_set);

%% ITI
% start trial with blank screen = baseline/pre-fixation epoch
set_bgcolor(bg_trial)
eventmarker(ec_iti_start)
idle(i ti)
eventmarker(ec_iti_end)

%% 1. FIXATION
% need to fixate centre (fixation breaks allowed). if no fixation acquired
% trial is stopped, error screen and timeout
run_scene(fixation_scene,ec_fix_screen_shown); 

    % no fixation
    if ~fixth.Success
            trialerror(ec_out_nofix)
            idle(nofix_timeout,bg_nofix)
            eventmarker(ec_timeout_screenshown)
    end
    

%% 2. OPTION 1 ONSET
% option 1 is shown. no fixation required to continue. record if/which cues
% are fixated.
if fixth.Success
TrialRecord.User.num_initiated_trials = TrialRecord.User.num_initiated_trials+1;

option1_onset_time = run_scene(option1_scene,ec_option1_on);
  
    if mag_fth1.Success
        eventmarker(ec_mag1_fix)
        if x_position1_i==1     % option 1 on the left
            if mag1_y_position_i==1     % mag is up
                eventmarker(ec_upleft_fix)
            else
                eventmarker(ec_downleft_fix)
            end
        else
            if mag1_y_position_i==1    
                eventmarker(ec_upright_fix)
            else
                eventmarker(ec_downright_fix)
            end
        end

        time_mag1_acquired = mag_fth1.AcquiredTime;                              

        % log RT and precise timings
        rt_mag1 = time_mag1_acquired - option1_onset_time;
        dvs_rts_mag1 = [dvs_rts_mag1 rt_mag1];
        dvs_ts_mag1_acquired = [dvs_ts_mag1_acquired time_mag1_acquired];
        dvs_mag1_held = [dvs_mag1_held fta_mag1.FixTime];
        bhv_variable('dvs_rts_mag1', dvs_rts_mag1);
        bhv_variable('dvs_ts_mag1_acquired', dvs_ts_mag1_acquired);
        bhv_variable('dvs_mag1_held',dvs_mag1_held)
    end

    if prob_fth1.Success
        eventmarker(ec_prob1_fix)
        if x_position1_i==1     % option 1 on the left
            if mag1_y_position_i==2     % mag is down so prob is up
                eventmarker(ec_upleft_fix)
            else
                eventmarker(ec_downleft_fix)
            end
        else
            if mag1_y_position_i==2    
                eventmarker(ec_upright_fix)
            else
                eventmarker(ec_downright_fix)
            end
        end

        time_prob1_acquired = prob_fth1.AcquiredTime;

        % log RT and precise timings
        rt_prob1 = time_prob1_acquired - option1_onset_time;
        dvs_rts_prob1 = [dvs_rts_prob1 rt_prob1];
        dvs_ts_prob1_acquired = [dvs_ts_prob1_acquired time_prob1_acquired];
        dvs_prob1_held = [dvs_prob1_held fta_prob1.FixTime];
        bhv_variable('dvs_rts_prob1', dvs_rts_prob1);
        bhv_variable('dvs_ts_prob1_acquired', dvs_ts_prob1_acquired);
        bhv_variable('dvs_prob1_held',dvs_prob1_held)
    end

%% 3. OPTION 2 ONSET
% option 2 is shown. no fixation required to continue.
    idle(100)
    option2_onset_time = run_scene(option2_scene,ec_option2_on);

    if mag_fth2.Success
        eventmarker(ec_mag2_fix)
        if x_position2_i==1     % option 2 on the left
            if mag1_y_position_i==1     % mag is up
                eventmarker(ec_upleft_fix)
            else
                eventmarker(ec_downleft_fix)
            end
        else
            if mag1_y_position_i==1    
                eventmarker(ec_upright_fix)
            else
                eventmarker(ec_downright_fix)
            end
        end

        time_mag2_acquired = mag_fth2.AcquiredTime;                              

        % log RT and precise timings
        rt_mag2 = time_mag2_acquired - option2_onset_time;
        dvs_rts_mag2 = [dvs_rts_mag2 rt_mag2];
        dvs_ts_mag2_acquired = [dvs_ts_mag2_acquired time_mag2_acquired];
        dvs_mag2_held = [dvs_mag2_held fta_mag2.FixTime];
        bhv_variable('dvs_rts_mag2', dvs_rts_mag2);
        bhv_variable('dvs_ts_mag2_acquired', dvs_ts_mag2_acquired);
        bhv_variable('dvs_mag2_held',dvs_mag2_held)
    end

    if prob_fth2.Success
        eventmarker(ec_prob2_fix)
        if x_position2_i==1     % option 1 on the left
            if mag1_y_position_i==2     % mag is down so prob is up
                eventmarker(ec_upleft_fix)
            else
                eventmarker(ec_downleft_fix)
            end
        else
            if mag1_y_position_i==2    
                eventmarker(ec_upright_fix)
            else
                eventmarker(ec_downright_fix)
            end
        end

        time_prob2_acquired = prob_fth2.AcquiredTime;

        % log RT and precise timings
        rt_prob2 = time_prob2_acquired - option2_onset_time;
        dvs_rts_prob2 = [dvs_rts_prob2 rt_prob2];
        dvs_ts_prob2_acquired = [dvs_ts_prob2_acquired time_prob2_acquired];
        dvs_prob2_held = [dvs_prob2_held fta_prob2.FixTime];
        bhv_variable('dvs_rts_prob2', dvs_rts_prob2);
        bhv_variable('dvs_ts_prob2_acquired', dvs_ts_prob2_acquired); 
        bhv_variable('dvs_prob2_held',dvs_prob2_held)
    end
    

     
    
%% 4. REACQUIRE FIXATION 
% need to fixate centre (fixation breaks allowed). if no fixation acquired
% trial is aborted, error screen and timeout
run_scene(fixation2_scene,ec_fix2_on);  

    % no fixation
    if ~fixth2.Success
            trialerror(ec_out_nofix)
            idle(nofix_timeout,bg_nofix)
            eventmarker(ec_out_nofix)
            eventmarker(ec_timeout_screenshown)
    end
    

%% 5. CHOICE
% binary choice between left and right
if fixth2.Success
    eventmarker(ec_fix2made)
    choice_onset_time = run_scene(choice_scene,ec_choice_on);   

    % no fixation
    if tc_ch.Success
        trialerror(ec_out_nochoice)
        idle(nochoice_timeout,bg_nofix) 
        eventmarker(ec_timeout_screenshown)
    end
    
    eventmarker(ec_choicemade)
    
    % choice made
    if fth3.Success     % chose option 1
        time_choice_acquired = fth3.AcquiredTime;  
        chosen_value = option1_value;
        unchosen_value = option2_value;
        chosen_mag  = mag1_i;
        chosen_prob = prob1_i;
        unchosen_mag  = mag2_i;
        unchosen_prob = prob2_i;
        chosen_option = [mag1 prob1];
        chosen_position = option1_x_position;
        chose_left = x_position1_i==1;
        chose_option1 = 1;
        
    elseif fth4.Success     % chose option 2
        time_choice_acquired = fth4.AcquiredTime; 
        chosen_value = option2_value;
        unchosen_value = option1_value;
        chosen_mag  = mag2_i;
        chosen_prob = prob2_i;
        unchosen_mag  = mag1_i;
        unchosen_prob = prob1_i;
        chosen_option = [mag2 prob2];
        chosen_position = option2_x_position;
        chose_left = x_position2_i==1;
        chose_option1 = 0;
        
    end

if fth3.Success || fth4.Success
    TrialRecord.User.block_trial_counter = TrialRecord.User.block_trial_counter+1;
    
    % log RT and precise timings
    rt_choice = time_choice_acquired - choice_onset_time;
    dvs_rts_choice = [dvs_rts_choice rt_choice];
    dvs_ts_choice_acquired = [dvs_ts_choice_acquired time_choice_acquired];
    bhv_variable('dvs_rts_choice', dvs_rts_choice);
    bhv_variable('dvs_ts_choice_acquired', dvs_ts_choice_acquired);
    TrialRecord.User.num_choicemade_trials = TrialRecord.User.num_choicemade_trials+1;
    
    dvs_chosen_value = [dvs_chosen_value chosen_value];
    dvs_chosen_mag = [dvs_chosen_mag chosen_mag];
    dvs_chosen_prob = [dvs_chosen_prob chosen_prob];
    dvs_unchosen_value = [dvs_unchosen_value unchosen_value];
    dvs_unchosen_mag = [dvs_unchosen_mag unchosen_mag];
    dvs_unchosen_prob = [dvs_unchosen_prob unchosen_prob];
    dvs_chose_left = [dvs_chose_left chose_left];
    dvs_chose_option1 = [dvs_chose_option1 chose_option1];
    bhv_variable('dvs_chosen_value',dvs_chosen_value);
    bhv_variable('dvs_chosen_mag',dvs_chosen_mag);
    bhv_variable('dvs_chosen_prob',dvs_chosen_prob);
    bhv_variable('dvs_unchosen_value',dvs_unchosen_value);
    bhv_variable('dvs_unchosen_mag',dvs_unchosen_mag);
    bhv_variable('dvs_unchosen_prob',dvs_unchosen_prob);
    bhv_variable('dvs_chose_left',dvs_chose_left);
    bhv_variable('dvs_chose_option1',dvs_chose_option1);

    % check fixation behaviour during choice epoch (what cues were looked
    % at)
    if mag_fth3.Success
        eventmarker(ec_mag1_fix)
        time_mag3_acquired = mag_fth3.AcquiredTime;                              

        % log RT and precise timings
        rt_mag3 = time_mag3_acquired - choice_onset_time;
        dvs_rts_mag3 = [dvs_rts_mag3 rt_mag3];
        dvs_ts_mag3_acquired = [dvs_ts_mag2_acquired time_mag2_acquired];
        dvs_mag3_held = [dvs_mag3_held fta_mag3.FixTime];
        bhv_variable('dvs_rts_mag3', dvs_rts_mag3);
        bhv_variable('dvs_ts_mag3_acquired', dvs_ts_mag3_acquired);
        bhv_variable('dvs_mag3_held',dvs_mag3_held)
    end

    if prob_fth3.Success
        eventmarker(ec_prob1_fix)
        time_prob3_acquired = prob_fth3.AcquiredTime;

        % log RT and precise timings
        rt_prob3 = time_prob3_acquired - choice_onset_time;
        dvs_rts_prob3 = [dvs_rts_prob3 rt_prob3];
        dvs_ts_prob3_acquired = [dvs_ts_prob3_acquired time_prob3_acquired];
        dvs_prob3_held = [dvs_prob3_held fta_prob3.FixTime];
        bhv_variable('dvs_rts_prob3', dvs_rts_prob3);
        bhv_variable('dvs_ts_prob3_acquired', dvs_ts_prob3_acquired); 
        bhv_variable('dvs_prob3_held',dvs_prob3_held)
    end

    if mag_fth4.Success
        eventmarker(ec_mag2_fix)
        time_mag4_acquired = mag_fth4.AcquiredTime;                              

        % log RT and precise timings
        rt_mag4 = time_mag4_acquired - choice_onset_time;
        dvs_rts_mag4 = [dvs_rts_mag4 rt_mag4];
        dvs_ts_mag4_acquired = [dvs_ts_mag4_acquired time_mag4_acquired];
        dvs_mag3_held = [dvs_mag4_held fta_mag4.FixTime];
        bhv_variable('dvs_rts_mag4', dvs_rts_mag4);
        bhv_variable('dvs_ts_mag4_acquired', dvs_ts_mag4_acquired);
        bhv_variable('dvs_mag4_held',dvs_mag4_held)
    end

    if prob_fth4.Success
        eventmarker(ec_prob2_fix)
        time_prob4_acquired = prob_fth4.AcquiredTime;

        % log RT and precise timings
        rt_prob4 = time_prob4_acquired - choice_onset_time;
        dvs_rts_prob4 = [dvs_rts_prob4 rt_prob4];
        dvs_ts_prob4_acquired = [dvs_ts_prob4_acquired time_prob4_acquired];
        dvs_prob4_held = [dvs_prob4_held fta_prob4.FixTime];
        bhv_variable('dvs_rts_prob4', dvs_rts_prob4);
        bhv_variable('dvs_ts_prob4_acquired', dvs_ts_prob4_acquired); 
        bhv_variable('dvs_prob4_held',dvs_prob4_held)
    end

%     
%% 5. FEEDBACK (reward) 
% reward is given whether correct choice is made or not. correct/incorrect choice signalled by bg colour. 
% only chosen option shown 

% will juice be delivered? based on chosen probability
roll = randi(1000);
if roll < chosen_prob*91
    reward_duration = chosen_mag*magnitude_reward_increments;
else 
    reward_duration = 0;
end

% set up remaining scene parameters
box5.Position = [chosen_position 0];

wth_chosen = WaitThenHold(f5);
wth_chosen.HoldTime = hold_target_time;
wth_chosen.WaitTime = reward_duration + 1000;   % if no reward, shown for 1s

con_chosen = Concurrent(wth_chosen);
con_chosen.add(box5)

% set background colour
if unchosen_value > chosen_value
    set_bgcolor(bg_wrong)
    trialerror(ec_out_incorrectchoice)
    TrialRecord.User.perc_correct_trials = (TrialRecord.User.num_correct_trials/TrialRecord.User.num_choicemade_trials)*100;
elseif unchosen_value <= chosen_value
    set_bgcolor(bg_correct)
    TrialRecord.User.num_correct_trials = TrialRecord.User.num_correct_trials+1;
    TrialRecord.User.perc_correct_trials = (TrialRecord.User.num_correct_trials/TrialRecord.User.num_choicemade_trials)*100;
    trialerror(ec_out_correctchoice)
end

% finally run scene
feedback_scene = create_scene(con_chosen,chosen_option);

% deliver juice while feedback scene happens
if reward_duration~=0
    goodmonkey(reward_duration,'eventmarker',ec_pump_on,'NonBlocking',2);   
    feedback_scene_onset = run_scene(feedback_scene,ec_fbscreenshown);      
    eventmarker(ec_pump_off)
else 
    feedback_scene_onset = run_scene(feedback_scene,ec_fbscreenshown);
    eventmarker(ec_unrewarded)
end


end
end
end


fprintf('Number trials initiated: %0.0f\n',TrialRecord.User.num_initiated_trials);
fprintf('Number trials attempted: %0.0f\n',TrialRecord.User.num_choicemade_trials);
fprintf('Performance: %0.0f%% correct\n',TrialRecord.User.perc_correct_trials);

%% Functions
% log cue ids
function log_option_ids(ec1,ec2,ec3)
    eventmarker(ec1)    % logging option 1 or option 2 next
    eventmarker(ec_optionids(ec2,ec3))
end

% log cue locations on screen
function log_cue_locs(ec1,ec2,ec3)
    eventmarker(ec1)    % logging position of cue 1 or cue 2 next
    eventmarker(ec2(ec3))
end



