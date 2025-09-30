# PEECOM Framework: Core Methodology Flow

## Simplified Framework Overview

This document outlines the **essential flow** of the PEECOM framework, showing how the major components connect from data input to final validation.

---

## 🎯 Core Methodology Flow (8 Steps)

### **Step 1: Data Input**
- **Component**: CMOHS Hydraulic Dataset
- **Details**: 2,205 samples, 54 features, 3 temporal blocks
- **Role**: Raw sensor data with hidden block-level artifacts

### **Step 2: Dual-Path Analysis** 
**Two parallel analyses of the same data:**

#### **Path A: Diagnostic Tools** 
- **Component**: RandomForest + Other Classifiers
- **Task**: Block prediction (try to guess which collection block each sample came from)
- **Purpose**: Detect if systematic artifacts exist

#### **Path B: PEECOM Testbed**
- **Component**: PEECOM Application Model  
- **Task**: Hydraulic classification (normal/degradation/fault)
- **Purpose**: The real model we want to protect and deploy

### **Step 3: Pre-Remediation Results**
#### **Path A Results**: Diagnostic Detection
- **Result**: 95.8% ± 2.1% block prediction accuracy
- **Interpretation**: SEVERE LEAKAGE detected (should be 33.3% if no artifacts)

#### **Path B Results**: PEECOM Performance  
- **Result**: High classification accuracy
- **Interpretation**: PEECOM is exploiting the same artifacts (unreliable!)

### **Step 4: BLAST Remediation** (Central Process)
- **Component**: Block normalization algorithm
- **Methods**: Mean correction + Covariance alignment
- **Purpose**: Remove systematic differences between blocks
- **Input**: Raw data with artifacts
- **Output**: Clean data without block signatures

### **Step 5: Post-Remediation Validation** 
**Re-test both paths on clean data:**

#### **Path A Validation**: Diagnostic Check
- **Result**: 33.3% ± 0.2% block prediction
- **Interpretation**: SUCCESS - no more block artifacts detectable

#### **Path B Validation**: PEECOM Protection
- **Result**: 33.2% ± 0.6% classification accuracy  
- **Interpretation**: SUCCESS - PEECOM now protected from artifacts

### **Step 6: Framework Success**
- **Evidence**: Both diagnostic and application models achieve chance-level performance
- **Meaning**: Artifacts eliminated, models no longer exploit collection biases

### **Step 7: Statistical Confirmation**
- **Methods**: Multi-seed validation, permutation testing, effect sizes
- **Results**: p-values > 0.05, Cohen's d ≈ 0
- **Confidence**: Statistically robust remediation success

### **Step 8: Universal Framework**
- **Impact**: Methodology applies to any temporal sensor ML application
- **Domains**: Medical devices, automotive, IoT, industrial monitoring

---

## 🔗 Key Connections Between Components

### **Data → Diagnostic Tools**
- **Connection**: Block prediction task
- **Method**: Train RandomForest to predict collection blocks
- **Purpose**: Quality control - detect if artifacts exist

### **Data → PEECOM Testbed** 
- **Connection**: Hydraulic classification task
- **Method**: Train PEECOM for actual application (fault detection)
- **Purpose**: The model we want to deploy reliably

### **Diagnostic Results → Remediation**
- **Connection**: Leakage detection triggers remediation
- **Logic**: IF diagnostic accuracy > chance THEN apply BLAST
- **Purpose**: Only remediate when artifacts are confirmed

### **PEECOM Results → Remediation**
- **Connection**: Application model also needs protection
- **Logic**: High performance likely means artifact exploitation
- **Purpose**: Protect the deployment model

### **Remediation → Both Paths**
- **Connection**: Same normalization applied to both diagnostic and application tasks
- **Method**: Block mean + covariance normalization
- **Purpose**: Eliminate artifacts while preserving legitimate patterns

### **Post-Remediation → Validation**
- **Connection**: Statistical confirmation of success
- **Methods**: Cross-validation, permutation testing
- **Criteria**: Chance-level performance + statistical insignificance

---

## 📊 Why This Flow Works

### **1. Dual-Role Design**
- **Diagnostic tools**: Detect problems (not for deployment)
- **PEECOM testbed**: The actual model we want to use (needs protection)
- **Separation**: Clear roles prevent confusion

### **2. Before-After Validation**
- **Before**: Both paths show high performance (suspicious)
- **After**: Both paths show chance performance (artifacts removed)
- **Proof**: Remediation worked for both diagnostic and application models

### **3. Universal Applicability** 
- **Diagnostic component**: Works for any temporal sensor data
- **Application component**: Can be any ML model (not just PEECOM)
- **Framework**: Methodology transfers across domains

---

## 🎨 Manual Flowchart Design Guide

If drawing manually, use this simplified structure:

```
[Data] 
   ↓
[Split into two paths]
   ↓                    ↓
[Diagnostic]    [PEECOM Testbed]
   ↓                    ↓  
[95.8% block]   [High accuracy]
   ↓                    ↓
      ↘            ↙
    [BLAST Remediation]
      ↙            ↘
   ↓                    ↓
[33.3% block]   [33.2% accuracy] 
   ↓                    ↓
      ↘            ↙
   [Framework Success]
```

### **Box Colors (if using colors):**
- **Purple**: Data input
- **Red**: Diagnostic tools  
- **Blue**: PEECOM testbed
- **Green**: Remediation
- **Yellow**: Validation/success

### **Key Arrows to Emphasize:**
1. Data splitting into two parallel paths
2. Both paths feeding into central remediation
3. Remediation feeding back to both paths
4. Both paths converging to framework success

---

## 🔑 Core Message

**"We use diagnostic tools to detect block leakage, then apply BLAST remediation to protect our application model (PEECOM) from these artifacts."**

This flow makes it clear that:
- Diagnostic classifiers are **tools for detection**, not the final model
- PEECOM is the **application model** we want to deploy
- BLAST is the **bridge** that protects application models from artifacts
- The framework is **universal** - any researcher can apply this flow

---

*This simplified flow focuses on the essential methodology without getting lost in technical details, making it perfect for flowchart visualization and manuscript clarity.*