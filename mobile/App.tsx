import { Ionicons } from "@expo/vector-icons";
import * as ImagePicker from "expo-image-picker";
import { StatusBar } from "expo-status-bar";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Image,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View
} from "react-native";

import { API_BASE_URL } from "./src/config";
import { ApiError, getModelInfo, healthCheck, predictImage } from "./src/api/dermascanApi";
import { colors, spacing } from "./src/theme";
import type { ModelInfo, PredictionResponse, ScanMetadata } from "./src/types";

const emptyMetadata: ScanMetadata = {
  age_range: "",
  sex: "",
  body_location: "",
  symptom_duration: ""
};

function formatConfidence(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export default function App() {
  const [isBackendReady, setIsBackendReady] = useState<boolean | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [selectedImage, setSelectedImage] = useState<ImagePicker.ImagePickerAsset | null>(null);
  const [metadata, setMetadata] = useState<ScanMetadata>(emptyMetadata);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const statusLabel = useMemo(() => {
    if (isBackendReady === null) {
      return "Checking API";
    }
    return isBackendReady ? "API online" : "API unavailable";
  }, [isBackendReady]);

  const refreshBackendStatus = useCallback(async () => {
    const [healthy, info] = await Promise.all([healthCheck(), getModelInfo()]);
    setIsBackendReady(healthy);
    setModelInfo(info);
  }, []);

  useEffect(() => {
    void refreshBackendStatus();
  }, [refreshBackendStatus]);

  const pickImage = async () => {
    setErrorMessage(null);
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setErrorMessage("Photo library permission is required to select an image.");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.9,
      allowsEditing: true,
      aspect: [1, 1]
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0]);
      setPrediction(null);
    }
  };

  const submitScan = async () => {
    if (!selectedImage) {
      setErrorMessage("Select a clear skin image before starting analysis.");
      return;
    }

    setIsAnalyzing(true);
    setErrorMessage(null);
    setPrediction(null);

    try {
      const response = await predictImage(
        {
          uri: selectedImage.uri,
          name: selectedImage.fileName ?? "skin-image.jpg",
          mimeType: selectedImage.mimeType ?? "image/jpeg"
        },
        metadata
      );
      setPrediction(response);
    } catch (error) {
      setErrorMessage(
        error instanceof ApiError
          ? error.message
          : "The analysis service is unavailable. Check that the backend is running."
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  const updateMetadata = (key: keyof ScanMetadata, value: string) => {
    setMetadata((current) => ({ ...current, [key]: value }));
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="dark" />
      <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : undefined} style={styles.keyboard}>
        <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
          <View style={styles.header}>
            <View>
              <Text style={styles.eyebrow}>DermaScan AI</Text>
              <Text style={styles.title}>Skin image review</Text>
            </View>
            <Pressable accessibilityRole="button" onPress={refreshBackendStatus} style={styles.iconButton}>
              <Ionicons name="refresh" size={20} color={colors.primaryDark} />
            </Pressable>
          </View>

          <View style={styles.statusRow}>
            <View style={[styles.statusDot, isBackendReady ? styles.onlineDot : styles.offlineDot]} />
            <Text style={styles.statusText}>{statusLabel}</Text>
            <Text style={styles.apiUrl} numberOfLines={1}>
              {API_BASE_URL}
            </Text>
          </View>

          {modelInfo ? (
            <View style={styles.notice}>
              <Text style={styles.noticeText}>
                {modelInfo.inference_mode === "placeholder"
                  ? "Development placeholder mode is active."
                  : `Model: ${modelInfo.model_name}`}
              </Text>
            </View>
          ) : null}

          <Pressable accessibilityRole="button" onPress={pickImage} style={styles.imagePicker}>
            {selectedImage ? (
              <Image source={{ uri: selectedImage.uri }} style={styles.previewImage} />
            ) : (
              <View style={styles.emptyImage}>
                <Ionicons name="image-outline" size={42} color={colors.primary} />
                <Text style={styles.emptyImageText}>Choose image</Text>
              </View>
            )}
          </Pressable>

          <View style={styles.formGrid}>
            <Field
              label="Age range"
              placeholder="Example: 25-34"
              value={metadata.age_range}
              onChangeText={(value) => updateMetadata("age_range", value)}
            />
            <Field
              label="Sex"
              placeholder="Optional"
              value={metadata.sex}
              onChangeText={(value) => updateMetadata("sex", value)}
            />
            <Field
              label="Body location"
              placeholder="Example: forearm"
              value={metadata.body_location}
              onChangeText={(value) => updateMetadata("body_location", value)}
            />
            <Field
              label="Duration"
              placeholder="Example: 2 weeks"
              value={metadata.symptom_duration}
              onChangeText={(value) => updateMetadata("symptom_duration", value)}
            />
          </View>

          {errorMessage ? (
            <View style={styles.errorBox}>
              <Ionicons name="alert-circle" size={18} color={colors.danger} />
              <Text style={styles.errorText}>{errorMessage}</Text>
            </View>
          ) : null}

          <Pressable
            accessibilityRole="button"
            disabled={isAnalyzing}
            onPress={submitScan}
            style={({ pressed }) => [styles.primaryButton, pressed && !isAnalyzing ? styles.buttonPressed : null]}
          >
            {isAnalyzing ? (
              <ActivityIndicator color="#ffffff" />
            ) : (
              <>
                <Ionicons name="scan" size={20} color="#ffffff" />
                <Text style={styles.primaryButtonText}>Analyze image</Text>
              </>
            )}
          </Pressable>

          {prediction ? <ResultsCard prediction={prediction} /> : null}

          <Text style={styles.disclaimer}>
            This app is educational only. It does not diagnose skin conditions. For symptoms, changes, pain,
            bleeding, fever, or concern, consult a qualified healthcare professional.
          </Text>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

type FieldProps = {
  label: string;
  placeholder: string;
  value: string;
  onChangeText: (value: string) => void;
};

function Field({ label, placeholder, value, onChangeText }: FieldProps) {
  return (
    <View style={styles.field}>
      <Text style={styles.fieldLabel}>{label}</Text>
      <TextInput
        autoCapitalize="none"
        onChangeText={onChangeText}
        placeholder={placeholder}
        placeholderTextColor="#8b9990"
        style={styles.input}
        value={value}
      />
    </View>
  );
}

function ResultsCard({ prediction }: { prediction: PredictionResponse }) {
  return (
    <View style={styles.results}>
      <View style={styles.resultsHeader}>
        <Text style={styles.resultsTitle}>
          {prediction.top_prediction ? prediction.top_prediction.class_name : "Uncertain result"}
        </Text>
        <Text style={styles.confidenceBadge}>{prediction.confidence_level}</Text>
      </View>

      {prediction.top_prediction ? (
        <Text style={styles.resultLead}>Top confidence: {formatConfidence(prediction.top_prediction.confidence)}</Text>
      ) : (
        <Text style={styles.resultLead}>The model could not identify a confident top category.</Text>
      )}

      <View style={styles.predictionList}>
        {prediction.top_k.map((item) => (
          <View key={item.class_name} style={styles.predictionRow}>
            <Text style={styles.predictionName}>{item.class_name}</Text>
            <Text style={styles.predictionValue}>{formatConfidence(item.confidence)}</Text>
          </View>
        ))}
      </View>

      <Text style={styles.sectionLabel}>Explanation</Text>
      <Text style={styles.bodyText}>{prediction.explanation.summary}</Text>
      <Text style={styles.sectionLabel}>Next steps</Text>
      <Text style={styles.bodyText}>{prediction.explanation.next_steps}</Text>
      <Text style={styles.warningText}>{prediction.explanation.warning}</Text>
      <Text style={styles.resultDisclaimer}>{prediction.disclaimer}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: colors.background
  },
  keyboard: {
    flex: 1
  },
  container: {
    padding: spacing.lg,
    gap: spacing.md
  },
  header: {
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "space-between"
  },
  eyebrow: {
    color: colors.primary,
    fontSize: 14,
    fontWeight: "700",
    letterSpacing: 0,
    textTransform: "uppercase"
  },
  title: {
    color: colors.ink,
    fontSize: 30,
    fontWeight: "800",
    letterSpacing: 0,
    marginTop: 2
  },
  iconButton: {
    alignItems: "center",
    backgroundColor: "#e7f3ed",
    borderRadius: 8,
    height: 44,
    justifyContent: "center",
    width: 44
  },
  statusRow: {
    alignItems: "center",
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    gap: spacing.sm,
    padding: spacing.md
  },
  statusDot: {
    borderRadius: 5,
    height: 10,
    width: 10
  },
  onlineDot: {
    backgroundColor: colors.success
  },
  offlineDot: {
    backgroundColor: colors.danger
  },
  statusText: {
    color: colors.ink,
    fontSize: 14,
    fontWeight: "700"
  },
  apiUrl: {
    color: colors.muted,
    flex: 1,
    fontSize: 12,
    textAlign: "right"
  },
  notice: {
    backgroundColor: "#fff6df",
    borderColor: "#ecd29c",
    borderRadius: 8,
    borderWidth: 1,
    padding: spacing.md
  },
  noticeText: {
    color: "#6e5620",
    fontSize: 14,
    fontWeight: "600"
  },
  imagePicker: {
    aspectRatio: 1,
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: 8,
    borderWidth: 1,
    overflow: "hidden"
  },
  previewImage: {
    height: "100%",
    width: "100%"
  },
  emptyImage: {
    alignItems: "center",
    flex: 1,
    gap: spacing.sm,
    justifyContent: "center"
  },
  emptyImageText: {
    color: colors.primaryDark,
    fontSize: 17,
    fontWeight: "700"
  },
  formGrid: {
    gap: spacing.md
  },
  field: {
    gap: spacing.xs
  },
  fieldLabel: {
    color: colors.ink,
    fontSize: 14,
    fontWeight: "700"
  },
  input: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: 8,
    borderWidth: 1,
    color: colors.ink,
    fontSize: 16,
    minHeight: 48,
    paddingHorizontal: spacing.md
  },
  errorBox: {
    alignItems: "center",
    backgroundColor: "#fff1f1",
    borderColor: "#e9b9b9",
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    gap: spacing.sm,
    padding: spacing.md
  },
  errorText: {
    color: colors.danger,
    flex: 1,
    fontSize: 14,
    fontWeight: "600"
  },
  primaryButton: {
    alignItems: "center",
    backgroundColor: colors.primary,
    borderRadius: 8,
    flexDirection: "row",
    gap: spacing.sm,
    justifyContent: "center",
    minHeight: 52,
    paddingHorizontal: spacing.lg
  },
  buttonPressed: {
    backgroundColor: colors.primaryDark
  },
  primaryButtonText: {
    color: "#ffffff",
    fontSize: 17,
    fontWeight: "800"
  },
  results: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: 8,
    borderWidth: 1,
    gap: spacing.md,
    padding: spacing.lg
  },
  resultsHeader: {
    alignItems: "flex-start",
    flexDirection: "row",
    gap: spacing.md,
    justifyContent: "space-between"
  },
  resultsTitle: {
    color: colors.ink,
    flex: 1,
    fontSize: 22,
    fontWeight: "800",
    textTransform: "capitalize"
  },
  confidenceBadge: {
    backgroundColor: "#e7f3ed",
    borderRadius: 8,
    color: colors.primaryDark,
    fontSize: 12,
    fontWeight: "800",
    overflow: "hidden",
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    textTransform: "uppercase"
  },
  resultLead: {
    color: colors.muted,
    fontSize: 15,
    fontWeight: "600"
  },
  predictionList: {
    borderTopColor: colors.border,
    borderTopWidth: 1
  },
  predictionRow: {
    alignItems: "center",
    borderBottomColor: colors.border,
    borderBottomWidth: 1,
    flexDirection: "row",
    justifyContent: "space-between",
    paddingVertical: spacing.sm
  },
  predictionName: {
    color: colors.ink,
    fontSize: 15,
    fontWeight: "700",
    textTransform: "capitalize"
  },
  predictionValue: {
    color: colors.primaryDark,
    fontSize: 15,
    fontWeight: "800"
  },
  sectionLabel: {
    color: colors.ink,
    fontSize: 13,
    fontWeight: "800",
    textTransform: "uppercase"
  },
  bodyText: {
    color: colors.muted,
    fontSize: 15,
    lineHeight: 22
  },
  warningText: {
    color: colors.danger,
    fontSize: 15,
    fontWeight: "700",
    lineHeight: 22
  },
  resultDisclaimer: {
    color: colors.muted,
    fontSize: 12,
    lineHeight: 18
  },
  disclaimer: {
    color: colors.muted,
    fontSize: 13,
    lineHeight: 20,
    textAlign: "center"
  }
});
