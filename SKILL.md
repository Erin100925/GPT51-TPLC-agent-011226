# Skill: fda_supply_chain_overview
**Description:** 針對醫療器材供應鏈（供應商裝箱單 + 醫院入庫清單）進行高層次總覽，包括主要流向、風險熱點、合規關注點與建議。
**Parameters:** 
- input_datasets: 供應商與醫院資料集（結構化文字或表格式摘要）
- focus: 可選的重點（如：風險、合規、營運效率）

# Skill: udi_and_label_validation
**Description:** 驗證裝置相關欄位是否符合 UDI 與標籤要求（例如：裝置識別碼、批號、效期格式、UDI segment 一致性）。
**Parameters:** 
- device_records: 包含 device_id、lot_number、expiry_date 等欄位的紀錄
- region: 目標法規區域（如：US-FDA, EU-MDR）

# Skill: lot_expiry_risk_scoring
**Description:** 依照批號與效期，結合出貨/入庫時間與數量，計算效期風險分數與潛在報廢風險。
**Parameters:** 
- transactions: 含 lot_number, expiry_date, ship_date, received_date, quantity 的交易紀錄
- horizon_days: 評估時間視窗（例如 30, 60, 90 天）

# Skill: shipment_receipt_reconciliation
**Description:** 比對供應商出貨與醫院入庫紀錄，找出數量差異、缺失收貨、重複收貨等異常。
**Parameters:** 
- supplier_data: 供應商裝箱單資料
- hospital_data: 醫院入庫清單資料
- tolerance: 數量容許誤差百分比或絕對值

# Skill: cold_chain_breach_detection
**Description:** 分析冷鏈相關裝置或溫度敏感產品，偵測運輸與存放過程中可能的溫度風險（若有溫度欄位或時間延遲）。
**Parameters:** 
- transactions: 含 ship_date, received_date, device_type 或溫度標記的紀錄
- max_transit_days: 運輸可接受最大天數
- risk_rules: 自訂風險規則（例如超出某時間即視為高風險）

# Skill: shortage_and_stockout_forecast
**Description:** 根據歷史出貨與入庫走勢預估短缺與缺貨風險，並提出備貨或再採購建議（高層級推估）。
**Parameters:** 
- time_series: 以時間序列彙總的消耗量或出貨量
- forecast_horizon: 預測期間長度（例如 30 天）
- safety_stock_policy: 安全存量政策簡述

# Skill: recall_impact_analysis
**Description:** 在模擬或實際召回情境下，評估特定批號或裝置召回對各醫院與病人安全的影響範圍。
**Parameters:** 
- transactions: 含 device_id, lot_number, hospital_name, quantity 的紀錄
- recalled_lots: 需召回的批號清單
- scenario_notes: 情境說明（例如「FDA 安全性通報」）

# Skill: data_quality_profiling
**Description:** 對供應鏈資料進行品質剖析，包含缺值率、欄位一致性、異常值與欄位格式問題，並給出修正建議。
**Parameters:** 
- dataset: 任一表格型資料
- critical_fields: 關鍵欄位列表（如 shipment_id, lot_number, quantity）

# Skill: anomaly_pattern_mining
**Description:** 找出異常模式，例如特定供應商常出現數量差異、特定醫院經常延遲入庫、特定裝置批號常有問題。
**Parameters:** 
- supplier_data: 供應商資料
- hospital_data: 醫院資料
- pattern_focus: 欲偵測的模式類型（數量異常、時間延遲、退貨率等）

# Skill: multi_site_comparison
**Description:** 跨醫院或跨供應商比較供應穩定性、異常比例、缺貨與效率，產出基準分析（benchmark）。
**Parameters:** 
- metrics_table: 已彙總的指標表（依 hospital_name 或 supplier_name）
- comparison_axis: 比較軸（醫院、供應商、地區）

# Skill: visualization_spec_generator
**Description:** 將分析需求轉換為視覺化規格建議（例如：圖表類型、維度、量測欄位、篩選器），並以自然語言描述。
**Parameters:** 
- analysis_goal: 分析目的與要回答的問題
- available_fields: 可用欄位與其意義

# Skill: markdown_graph_storyteller
**Description:** 根據供應鏈與風險分析結果，撰寫以 Markdown 格式呈現的圖表說明與視覺敘事文字。
**Parameters:** 
- findings: 主要數據發現與指標
- graph_plan: 預計製作的圖表清單與目的

# Skill: regulatory_gap_assessment
**Description:** 針對 FDA 21 CFR 820（QSR）、Part 11 等條文，從流程與資料角度說明目前差距與改善建議。
**Parameters:** 
- process_description: 目前流程文字敘述
- evidence_snippets: 來自資料分析的重點或案例

# Skill: narrative_report_writer
**Description:** 將技術分析與圖表結果轉換成面向管理層或稽核單位可讀的長篇報告（可中英雙語），強調風險與行動方案。
**Parameters:** 
- audience: 目標讀者（例如 品管主管、RA、FDA 稽核）
- key_points: 必須涵蓋的重點列表
- language: 語言（zh-TW, en 或 bilingual）

# Skill: prompt_chain_planner
**Description:** 根據當前需求，設計多代理、多步驟 Prompt 鏈，指定每一代理輸入與輸出以及交接方式。
**Parameters:** 
- goal: 最終想要達成的分析或文件成果
- available_agents: 可用代理與其能力簡述
- constraints: 限制條件（如時間、成本、模型限制）
