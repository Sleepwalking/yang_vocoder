function output = ...
  SynthesizeByEventBasedMethod(event_index, event_time, ...
                               harmonics_level_at_event, ...
                               harmonics_deviation_at_event, fs, f0i, ...
                               unvoiced_mask)
% Synthesize speech from event based information made from lean
% representation
% output = ...
%  SynthesizeByEventBasedMethod(event_index, event_time, ...
%                               harmonics_level_at_event, ...
%                               harmonics_deviation_at_event, fs, f0i, ...
%                               unvoiced_mask)
%
% output = ...
%  SynthesizeByEventBasedMethod(event_index, event_time, ...
%                               harmonics_level_at_event, ...
%                               harmonics_deviation_at_event, fs, f0i, ...
%                               unvoiced_mask, interp_method)
%
% Input argument
%
% event_index: event location in sample index
% event_time: actual event location (s)
% harmonics_level_at_event: power information (dB)
% harmonics_deviation_at_event: rondom relative power information (dB)
% fs: sampling frequency (Hz)
% f0i: fundamental frequency in sample time (Hz)
% unvoiced_mask: silent indicator (for each event)
%
% Return value
%
% output: synthesized signal

% Copyright 2016 Google Inc. All Rights Reserved
% Author: hidekik@google.com (Hideki Kawahara)

%% convert to FFT-based array
narginchk(7, 7);
[response_for_periodic_trim, response_for_random_trim, ola_fftl] = ...
  GenerateResponses(f0i, fs, harmonics_level_at_event, ...
                    harmonics_deviation_at_event, event_time, event_index);
%%
out_buffer = f0i*0;
noise_out_buffer = out_buffer;
synth_out_buffer = out_buffer;
synth_noise_buffer = out_buffer;
nData = length(out_buffer);
half_width_of_BLIT = 2 * fs / (0.55 * fs);
blit_base = -ceil(half_width_of_BLIT):ceil(half_width_of_BLIT);
initial_index = 1;
noise_spread = max(1,min(1,unvoiced_mask(:))); % 0.2 to 1
base_noise = randn(length(out_buffer),1);
periodic_working_buffer = zeros(ola_fftl, 1);
zero_vector = periodic_working_buffer;
ola_zero = round(fs / min(f0i)) + 1;
ola_base = (1:ola_fftl)' - ola_zero;
sample_index = (1:nData)';
for ii = 1:length(event_time)
  base_index = event_index(ii) - blit_base; % minus (-) is the proper sign
  [synth_out_buffer, out_buffer] = ...
    UpdatePeriodicComponent(response_for_periodic_trim, synth_out_buffer, ...
                            base_index, ii, zero_vector, event_time, ...
                            event_index, fs, ola_zero, ola_base, ola_fftl, ...
                            sample_index, unvoiced_mask, out_buffer);
  [synth_noise_buffer, noise_out_buffer, initial_index] = ...
    UpdateRandomComponent(response_for_random_trim, synth_noise_buffer, ...
                          initial_index, ii, noise_spread, event_time, ...
                          event_index, ola_zero, ola_base, ola_fftl, ...
                          sample_index, base_noise, noise_out_buffer);
end;
raw_blit = GenerateBlitImpulse(blit_base, half_width_of_BLIT);
blit_equalizer_response = DesignBlitEqualizer(raw_blit);
half_width_of_EQBLIT = round((length(blit_equalizer_response) - 1) / 2);
tmp_period = fftfilt(blit_equalizer_response, ...
                     [synth_out_buffer; zeros(2 * half_width_of_EQBLIT, 1)]);
synth_noise_out = synth_noise_buffer ./ sqrt(f0i);
synth_periodic_out = ...
  tmp_period(half_width_of_EQBLIT + (1:nData)) ./ sqrt(f0i);
output = synth_periodic_out + synth_noise_out;
end

function [response_for_periodic_trim, response_for_random_trim, ola_fftl] ...
  = GenerateResponses(f0i, fs, harmonics_level_at_event, ...
                      harmonics_deviation_at_event, event_time, event_index)
kResidualToSNR = 11; % calibration constant from residual to SNR (dB)
spectrum_out.used_f0 = f0i(event_index);
spectrum_out.sampling_frequency = fs;
spectrum_out.temporal_positions = event_time;
spectrum_out.harmonic_power_dB = harmonics_level_at_event;
spectral_envelope = CalculateSpectrumEnvelope(spectrum_out);
noise_out = spectrum_out;
noise_out.refined_f0 = f0i(event_index);
noise_out.frame_time = event_time;
noise_out.aperiodicity_matrix = harmonics_deviation_at_event;
aperiodicity_dB = ...
  CalculateAperiodicitySgram(noise_out) + kResidualToSNR;
aperiodicity_ratio = ConvertDecibelToPower(min(0, aperiodicity_dB));
maximum_response_length = 0.03; % default 30 ms
response_lengh = round(maximum_response_length * fs); % samples
maximum_noise_length = 2 / min(f0i);
noise_length = round(maximum_noise_length * fs); % samples
ola_fftl = 2 ^ ceil(log2(noise_length + response_lengh + 1));
fftl = (size(spectral_envelope ,1) - 1) * 2;

data_frequency_axis = (0:fftl - 1) / fftl * fs;
data_frequency_axis(data_frequency_axis > fs / 2) = ...
  data_frequency_axis(data_frequency_axis > fs / 2) - fs;
% TODO(hidekik) need check: originally 50 maybe 150 better? (hz)
% 1000 is the corner frequenccy (Hz)
low_noise_masker = ...
  GetSigmoidNoiseShaper(data_frequency_axis, 1000, 150);
spectral_envelope_DFTBIN = ...
  [spectral_envelope;spectral_envelope(end-1:-1:2,:)];
random_envelope_DFTBIN = ...
  [aperiodicity_ratio;aperiodicity_ratio(end-1:-1:2,:)];
random_envelope_SHAPED = ...
  diag(low_noise_masker) * random_envelope_DFTBIN;
periodic_part = max(0.0001, (1 - random_envelope_SHAPED));
periodic_spectrum = sqrt(periodic_part .* spectral_envelope_DFTBIN);
random_spectrum = ...
  sqrt(random_envelope_SHAPED .* spectral_envelope_DFTBIN);
%%
time_axis = (0:fftl -1)' / fs;
time_shaper = time_axis * 0 + 1;
time_shaper(time_axis > maximum_response_length) = 0;
time_segment = ...
  (time_axis(time_axis > 0.8 * maximum_response_length & ...
  time_axis <= maximum_response_length) ...
  -0.8 * maximum_response_length) / (0.2 * maximum_response_length);
time_shaper(time_axis > 0.8 * maximum_response_length & ...
  time_axis <= maximum_response_length) ...
  = 0.5 + 0.5 * cos(pi * time_segment);
cepstrum_for_periodic = ifft(log(periodic_spectrum));
complex_cepstrum_for_periodic = cepstrum_for_periodic;
complex_cepstrum_for_periodic(fftl / 2 + 1:end,:) = 0;
complex_cepstrum_for_periodic(2:fftl / 2,:) = ...
  complex_cepstrum_for_periodic(2:fftl / 2,:) * 2;
response_for_periodic = diag(time_shaper) * ...
  real(ifft(exp(fft(complex_cepstrum_for_periodic))));
response_for_periodic_trim = response_for_periodic(1:response_lengh, :);
cepstrum_for_random = ifft(log(random_spectrum));
complex_cepstrum_for_random = cepstrum_for_random;
complex_cepstrum_for_random(fftl / 2 + 1:end,:) = 0;
complex_cepstrum_for_random(2:fftl / 2,:) = ...
  complex_cepstrum_for_random(2:fftl / 2,:) * 2;
response_for_random = diag(time_shaper) * ...
  real(ifft(exp(fft(complex_cepstrum_for_random))));
response_for_random_trim = response_for_random(1:response_lengh, :);
end

function [synth_out_buffer, out_buffer] = ...
  UpdatePeriodicComponent(response_for_periodic_trim, synth_out_buffer, ...
                          base_index, ii, zero_vector, event_time, ...
                          event_index, fs, ola_zero, ola_base, ola_fftl, ...
                          sample_index, unvoiced_mask, out_buffer)
%
nData = length(out_buffer);
half_width_of_BLIT = 2 * fs / (0.55 * fs);
blit_base = -ceil(half_width_of_BLIT):ceil(half_width_of_BLIT);
periodic_working_buffer = zero_vector;
fractional_index = event_time(ii) * fs - event_index(ii) + 1 + blit_base;
tmp_blit = GenerateBlitImpulse(fractional_index, half_width_of_BLIT);
periodic_working_buffer(ola_zero - blit_base) = tmp_blit / sum(tmp_blit);
periodic_working_out_buffer = ...
  real(ifft(fft(periodic_working_buffer) .* ...
            fft(response_for_periodic_trim(:, ii), ola_fftl)));
out_buffer(max(1,min(nData,base_index))) = tmp_blit / sum(tmp_blit);
past_index_condition = event_index(ii) + ola_base > 0 & ...
  event_index(ii) + ola_base < nData;
synth_out_buffer(sample_index(event_index(ii) + ...
  ola_base(past_index_condition))) = ...
  periodic_working_out_buffer(past_index_condition) * ...
  (1 - unvoiced_mask(ii)) + ...
  synth_out_buffer(sample_index(event_index(ii) + ...
  ola_base(past_index_condition)));
end

function [synth_noise_buffer, noise_out_buffer, initial_index] = ...
  UpdateRandomComponent(response_for_random_trim, synth_noise_buffer, ...
                        initial_index, ii, noise_spread, event_time, ...
                        event_index, ola_zero, ola_base, ola_fftl, ...
                        sample_index, base_noise, noise_out_buffer)
%----
nData = length(noise_out_buffer);
prev_index = initial_index:event_index(ii);
if ii == length(event_time)
  post_index = event_index(ii):nData;
else
  post_index = event_index(ii):event_index(ii + 1);
end;
prev_time = (prev_index(:) - event_index(ii)) / length(prev_index);
post_time = (post_index(:) - event_index(ii)) / length(post_index);
noise_shape = 0.5 * [(1 + cos(pi*prev_time / noise_spread(ii))) .* ...
  (abs(prev_time / noise_spread(ii)) <= 1); ...
  (1 + cos(pi * post_time(2:end) / noise_spread(ii))) .* ...
  (abs(post_time(2:end) / noise_spread(ii)) <= 1)];
noise_out_buffer(prev_index(1):post_index(end)) = ...
  noise_out_buffer(prev_index(1):post_index(end)) + noise_shape .* ...
  base_noise(prev_index(1):post_index(end)) / sqrt(sum(noise_shape .^ 2));
initial_index = event_index(ii);
noise_base_index = (prev_index(1):post_index(end))' - event_index(ii);
noise_working_buffer(max(1,ola_zero + noise_base_index)) = ...
  noise_shape .* base_noise(prev_index(1):post_index(end)) / ...
  sqrt(sum(noise_shape .^ 2));
noise_working_out_buffer = ...
  real(ifft(fft(noise_working_buffer(:), ola_fftl) .* ...
            fft(response_for_random_trim(:, ii), ola_fftl)));
past_index_condition = event_index(ii) + ola_base > 0 & ...
  event_index(ii) + ola_base < nData;
synth_noise_buffer(sample_index(event_index(ii) + ...
  ola_base(past_index_condition))) = ...
  noise_working_out_buffer(past_index_condition) + ...
  synth_noise_buffer(sample_index(event_index(ii) + ...
  ola_base(past_index_condition)));
end
