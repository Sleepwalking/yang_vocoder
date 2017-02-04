function output = ...
  FindBestTrajectory(probability_structure, median_f0, tracking_condition)
% Find best path using the total probability map
%
% output = ...
%   FindBestTrajectory(probability_structure, median_f0)
%
% output = ...
%   FindBestTrajectory(probability_structure, median_f0, tracking_condition)
%

% Input argument
%
% probability_structure: structure with following fields, which are used
%   observation_probability_map : probability time series of each channel
%   amplitude_map : amplitude of filter output (absolute value)
%   center_frequency_list : list of center frequencies of filter (Hz)
%   frame_shift : frame shift (s)
%   channels_in_octave : detector channels per octave
%   temporal_positions : center location of the analysis window (s)
% median_f0 : median_f0 (Hz)
% tracking_condition : structure with following fields
%   peak_spread : spread in transition
%   deviation_spread : weighting from median F0
%   focus_spread : weighting from selecting peak in the original map
%
% Return value
%   output : structure with following fields
%     trajectory : best channel sequence
%     best_score_trace : score sequence on the best channel sequence
%     best_channel : internal use (best trajectory on the smoothed map)
%     median_channel : initial estimate of F0 median channel
%     median_index : integer representation of median_channel
%     elapsed_time : time needed to process (s)

% Copyright 2016 Google Inc. All Rights Reserved
% Author: hidekik@google.com (Hideki Kawahara)

start_tic = tic;
narginchk(2, 3)
if nargin == 2
    tracking_condition = DefaultTrackingOptions;
end;
peak_spread = tracking_condition.peak_spread;
deviation_spread = tracking_condition.deviation_spread;
focus_spread = tracking_condition.focus_spread;

prob_map_original = probability_structure.observation_probability_map;
amp_map = probability_structure.amplitude_map;
fc_list = probability_structure.center_frequency_list;
frame_shift = probability_structure.frame_shift;
n_frames = size(amp_map,2);
channels_in_octave = probability_structure.channels_in_octave;
[~, median_channel] = min(abs(fc_list - median_f0));
n_states = size(prob_map_original,1);

%%  convert to smoothed probability
smoothing_wodth = 0.045;
half_length = ceil(smoothing_wodth / frame_shift);
smoother = hanning(2 * half_length + 1);
rms_amp = sqrt(mean(amp_map .^ 2));
filler = randn(1, 4 * half_length) .^2 * mean(rms_amp) / 1000;
prob_map = prob_map_original;
smoothed_amp = fftfilt(smoother, [rms_amp, filler]);
for ii = 1:n_states
    tmp = fftfilt(smoother, [prob_map_original(ii, :) .* rms_amp, filler]);
    tmp = tmp ./ smoothed_amp;
    prob_map(ii, :) = tmp(half_length + (1:n_frames));
end;
peak_map = prob_map >= prob_map([1 1:end - 1],:) & ...
    prob_map > prob_map([2:end end],:);
original_peak_map = prob_map_original >= prob_map_original([1 1:end - 1],:) & ...
    prob_map_original > prob_map_original([2:end end],:);
[~, ac_channel] = min(abs(fc_list - 60));
% --- safe peak fill
peak_map(round(median_channel), sum(peak_map) < 1) = 1;
original_peak_map(round(median_channel), sum(original_peak_map) < 1) = 1;
%% --- node structure for tracking
state_list = 1:n_states;
node_str = ...
  struct('n_peaks', [], 'n_peaks_original', [], 'peak_channel', [], ...
         'original_peak_channel', [], 'probability', [], ...
         'original_probability', [], 'score', [], 'origin', []);
for ii = 1:n_frames
    s = struct('n_peaks', sum(peak_map(:, ii)), ...
               'n_peaks_original', sum(original_peak_map(:, ii)), ...
               'peak_channel', state_list(peak_map(:, ii) > 0.5), ...
               'original_peak_channel', ...
               state_list(original_peak_map(:, ii) > 0.5), ...
               'probability', prob_map(peak_map(:, ii) > 0.5, ii), ...
               'original_probability', ...
               prob_map_original(original_peak_map(:, ii) > 0.5, ii), ...
               'score', prob_map(peak_map(:, ii) > 0.5, ii), ... %initial state
               'origin', 1); % initial state
    node_str(ii) = s;
end;
%% forward track

sigma_peak = channels_in_octave * peak_spread;
sigma_deviation = channels_in_octave * deviation_spread;
for frame_id = 2:n_frames
    % these are row vectors
    past_channel = node_str(frame_id - 1).peak_channel;
    current_channel = node_str(frame_id).peak_channel;
    % the following code seems to run even slower because # of peaks is usually small
    % (except if you're a fan of automatic broadcast, drop repmat and see if it's faster)
    %proximity_score = ... % size = current x past
    %  exp(-((repmat(current_channel', 1, length(past_channel)) - ...
    %    repmat(past_channel, length(current_channel), 1)) / sigma_peak) .^ 2);
    deviation_score = exp(-((current_channel - median_channel) / ...
      sigma_deviation) .^ 2);
    %contribution_score = proximity_score .* ...
    %  repmat(node_str(frame_id - 1).score', length(current_channel), 1) .* ...
    %  repmat(deviation_score', 1, length(past_channel));
    for current_peak_id = 1:node_str(frame_id).n_peaks
        proximity_score = ...
          exp(-((current_channel(current_peak_id) - past_channel) / sigma_peak) .^ 2);
        contribution_score = proximity_score .* node_str(frame_id - 1).score' ...
          * deviation_score(current_peak_id);
        [max_value, max_channel_id] = max(contribution_score);
        node_str(frame_id).score(current_peak_id) = ...
            max_value * node_str(frame_id).probability(current_peak_id) * ...
            ((node_str(frame_id).peak_channel(current_peak_id) > ...
            ac_channel) + 0.000001);
        node_str(frame_id).origin(current_peak_id) = max_channel_id;
    end;
    node_str(frame_id).score = node_str(frame_id).score / ...
      sum(node_str(frame_id).score);
end;
%% back track
best_channel = zeros(n_frames, 1);
[~, max_peak_id] = max(node_str(n_frames).score);
best_channel(n_frames) = node_str(n_frames).peak_channel(max_peak_id);
for frame_id = n_frames - 1:-1:1 % n_frames -10
    best_peak_id = node_str(frame_id + 1).origin(max_peak_id);
    best_channel(frame_id) = node_str(frame_id).peak_channel(best_peak_id);
    max_peak_id = best_peak_id;
end;

%% make trajectory
sigma_focus = channels_in_octave * focus_spread;
trajectory = best_channel * 0;
best_score_trace = trajectory;
for frame_id = 1:n_frames
    temp_score = node_str(frame_id).original_probability' .* ...
        exp(-((node_str(frame_id).original_peak_channel - ...
            best_channel(frame_id)) / sigma_focus) .^ 2);
    [best_score_trace(frame_id), best_channel_id] = max(temp_score);
    trajectory(frame_id) = ...
      node_str(frame_id).original_peak_channel(best_channel_id);
end;

%%
output = struct('trajectory', trajectory, ...
                'best_score_trace', best_score_trace, ...
                'best_channel', best_channel, ...
                'median_channel', median_channel, ...
                'median_index', round(median_channel), ...
                'elapsed_time', toc(start_tic));
end
