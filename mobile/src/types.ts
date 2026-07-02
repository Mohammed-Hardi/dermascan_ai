export type PredictionItem = {
  class_name: string;
  confidence: number;
};

export type Explanation = {
  summary: string;
  next_steps: string;
  warning: string;
};

export type PredictionResponse = {
  scan_id: string;
  status: string;
  top_prediction: PredictionItem | null;
  top_k: PredictionItem[];
  confidence_level: string;
  risk_level: string;
  explanation: Explanation;
  disclaimer: string;
  model_version: string;
  created_at: string;
};

export type ModelInfo = {
  model_name: string;
  version: string;
  classes: string[];
  input_size: number[];
  is_placeholder: boolean;
  is_smoke_test: boolean;
  model_available: boolean;
  inference_mode: string;
  metrics: Record<string, number | null>;
  last_trained: string | null;
};

export type ScanMetadata = {
  age_range: string;
  sex: string;
  body_location: string;
  symptom_duration: string;
};
