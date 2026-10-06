// laya-cli: answer Jev-style requests with a Laya model.
//
//   laya-cli -m models/gguf/laya.gguf -f request.json
//   echo '{"state": {...}, "questions": {...}}' | laya-cli -m models/gguf/laya.gguf
//   laya-cli -m models/gguf/laya.gguf --jsonl requests.jsonl   # one request per line, one response per line
//
// Request:  {"state": <string | JSON>, "questions": {"<id>": {"type": "choice"|"score"|"noul",
//                                                          "instructions": ..., "criteria": ...}}}
// Response: the same shape as rl_agent_api.RLAgent.system_one().

#include "laya.h"
#include "tokenizer.h"
#include "laya-model.h"
#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;


static void py_escape(const std::string & s, bool ensure_ascii, std::string & out) {
    static const char * hex = "0123456789abcdef";
    auto u16 = [&](unsigned v) {
        out += "\\u";
        out += hex[(v >> 12) & 15];
        out += hex[(v >> 8) & 15];
        out += hex[(v >> 4) & 15];
        out += hex[v & 15];
    };
    out += '"';
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char) s[i];
        if (c < 0x80) {
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                default:
                    if (c < 0x20 || (ensure_ascii && c == 0x7f)) {
                        u16(c);
                    } else {
                        out += (char) c;
                    }
            }
            ++i;
            continue;
        }
        const int n = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : 2;
        if (!ensure_ascii) {
            out.append(s, i, n);
        } else {
            unsigned cp = c & (n == 4 ? 0x07 : n == 3 ? 0x0F : 0x1F);
            for (int j = 1; j < n && i + j < s.size(); ++j) {
                cp = (cp << 6) | ((unsigned char) s[i + j] & 0x3F);
            }
            if (cp >= 0x10000) {
                cp -= 0x10000;
                u16(0xD800 + (cp >> 10));
                u16(0xDC00 + (cp & 0x3FF));
            } else {
                u16(cp);
            }
        }
        i += n;
    }
    out += '"';
}

static void py_dumps(const json & v, bool ensure_ascii, std::string & out) {
    if (v.is_object()) {
        out += '{';
        bool first = true;
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (!first) out += ", ";
            first = false;
            py_escape(it.key(), ensure_ascii, out);
            out += ": ";
            py_dumps(it.value(), ensure_ascii, out);
        }
        out += '}';
    } else if (v.is_array()) {
        out += '[';
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) out += ", ";
            py_dumps(v[i], ensure_ascii, out);
        }
        out += ']';
    } else if (v.is_string()) {
        py_escape(v.get<std::string>(), ensure_ascii, out);
    } else if (v.is_boolean()) {
        out += v.get<bool>() ? "true" : "false";
    } else if (v.is_null()) {
        out += "null";
    } else {
        out += v.dump(); 
    }
}

static std::string py_dumps(const json & v, bool ensure_ascii) {
    std::string out;
    py_dumps(v, ensure_ascii, out);
    return out;
}

static std::string py_str(const json & v) {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_null()) return "None";
    if (v.is_boolean()) return v.get<bool>() ? "True" : "False";
    return v.dump();
}

static double round4(double x) {
    return std::round(x * 10000.0) / 10000.0;
}


struct question_storage {
    laya_question            q{};
    std::vector<std::string> keys, descs;
    std::vector<const char *> key_ptrs, desc_ptrs;
    std::vector<bool>        desc_null;
    std::string              instructions;
};

static bool parse_question(const std::string & qid, const json & qd, question_storage & s, std::string & err) {
    const std::string t = qd.value("type", "");
    s.instructions      = qd.contains("instructions") && qd["instructions"].is_string()
                              ? qd["instructions"].get<std::string>()
                              : py_dumps(qd.value("instructions", json()), true);
    const json crit     = qd.value("criteria", json());

    if (t == "choice") {
        s.q.type = LAYA_QTYPE_CHOICE;
        if (crit.is_array()) {
            for (const auto & c : crit) {
                s.keys.push_back(py_str(c));
                s.descs.emplace_back();
                s.desc_null.push_back(true);
            }
        } else if (crit.is_object()) {
            for (auto it = crit.begin(); it != crit.end(); ++it) {
                s.keys.push_back(it.key());
                const bool empty = it.value().is_null() || (it.value().is_string() && it.value().get<std::string>().empty());
                s.descs.push_back(empty ? "" : py_str(it.value()));
                s.desc_null.push_back(empty);
            }
        } else {
            err = "question '" + qid + "': choice criteria must be a list or an object";
            return false;
        }
    } else if (t == "score") {
        s.q.type = LAYA_QTYPE_SCORE;
        if (!crit.is_array()) {
            err = "question '" + qid + "': score criteria must be a list of level descriptions";
            return false;
        }
        for (const auto & c : crit) {
            s.keys.emplace_back();
            s.descs.push_back(py_str(c));
            s.desc_null.push_back(false);
        }
    } else if (t == "noul") {
        s.q.type = LAYA_QTYPE_NOUL;
        if (crit.is_object()) {
            for (const char * k : { "false", "true" }) {
                const json v     = crit.value(k, json());
                const bool empty = v.is_null() || (v.is_string() && v.get<std::string>().empty());
                s.keys.emplace_back(k);
                s.descs.push_back(empty ? "" : py_str(v));
                s.desc_null.push_back(empty);
            }
        }
    } else {
        err = "question '" + qid + "': unknown type '" + t + "'";
        return false;
    }
    for (size_t i = 0; i < s.keys.size(); ++i) {
        s.key_ptrs.push_back(s.keys[i].c_str());
        s.desc_ptrs.push_back(s.desc_null[i] ? nullptr : s.descs[i].c_str());
    }
    s.q.instructions  = s.instructions.c_str();
    s.q.keys          = s.key_ptrs.data();
    s.q.descriptions  = s.desc_ptrs.data();
    s.q.n_options     = (int32_t) s.keys.size();
    return true;
}

static json handle_request(laya_context * ctx, const laya_model * model, const json & req, bool with_logits) {
    const json &      st    = req.contains("state") ? req["state"] : json("");
    const std::string state = st.is_string() ? st.get<std::string>() : py_dumps(st, false);

    json answers = json::object();
    int  n_tok   = 0;
    for (auto it = req["questions"].begin(); it != req["questions"].end(); ++it) {
        const std::string & qid = it.key();
        question_storage    s;
        std::string         err;
        if (!parse_question(qid, it.value(), s, err)) {
            throw std::runtime_error(err);
        }
        const int          n_max = std::max<int>(2, s.q.n_options);
        std::vector<float> probs(n_max), logits(n_max);
        laya_answer a{};
        if (laya_run_inference(ctx, state.c_str(), &s.q, probs.data(), logits.data(), &a) != 0) {
            return 1;
        }
        n_tok += a.n_tokens;
        json ext = { { "act_probability", (double) a.act_probability } };
        if (with_logits) {
            ext["logits"]      = std::vector<double>(logits.begin(), logits.begin() + a.n_options);
            ext["temperature"] = a.temperature;
        }
        json ans;
        if (s.q.type == LAYA_QTYPE_CHOICE) {
            json p = json::object();
            for (int i = 0; i < a.n_options; ++i) p[s.keys[i]] = round4(probs[i]);
            ans = { { "type", "choice" }, { "choice", s.keys[a.best] }, { "probabilities", p },
                    { "confidence", round4(a.confidence) }, { "rl_agent", ext } };
        } else if (s.q.type == LAYA_QTYPE_SCORE) {
            json legend = json::object(), p = json::object();
            for (int i = 0; i < a.n_options; ++i) {
                legend[std::to_string(i)] = s.descs[i];
                p[std::to_string(i)]      = round4(probs[i]);
            }
            ans = { { "type", "score" }, { "score", round4(a.value) }, { "legend", legend }, { "probabilities", p },
                    { "confidence", round4(a.confidence) }, { "rl_agent", ext } };
        } else {
            ans = { { "type", "noul" }, { "noul", round4(a.value) }, { "rl_agent", ext } };
        }
        answers[qid] = ans;
    }
    return { { "model", laya_model_name(model) }, { "answers", answers },
             { "usage", { { "input_tokens", n_tok }, { "output_tokens", 0 } } } };
}

static void usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s -m MODEL.gguf [options]\n"
            "  -m, --model PATH      Laya GGUF (encoder, decision head and tokenizer in one file)\n"
            "  -f, --file PATH       request JSON (default: stdin)\n"
            "      --jsonl           input has one request per line; print one response per line\n"
            "  -t, --threads N       CPU threads (default: half the hardware threads)\n"
            "      --logits          include raw logits and temperature in the output\n"
            "      --tokenize TEXT   print the token ids of TEXT and exit\n"
            "      --sequence        print the encoder input ids / markers per question instead of answering\n"
            "      --bench N         run every request N times and report timing to stderr\n"
            "      --no-mmap         read the weights into memory instead of mapping the file\n"
            "  -v, --verbose         show ggml / llama.cpp logs\n",
            argv0);
}

int main(int argc, char ** argv) {
    std::string model_path, file, tokenize_text;
    bool        jsonl = false, logits = false, verbose = false, tokenize = false, sequence = false, no_mmap = false;
    int         threads = 0, bench = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a    = argv[i];
        auto              next = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage(argv[0]);
                exit(1);
            }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") model_path = next();
        else if (a == "-f" || a == "--file") file = next();
        else if (a == "-t" || a == "--threads") threads = std::stoi(next());
        else if (a == "--jsonl") jsonl = true;
        else if (a == "--logits") logits = true;
        else if (a == "--tokenize") { tokenize = true; tokenize_text = next(); }
        else if (a == "--sequence") sequence = true;
        else if (a == "--bench") bench = std::stoi(next());
        else if (a == "--no-mmap") no_mmap = true;
        else if (a == "-v" || a == "--verbose") verbose = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown argument: %s\n", a.c_str()); usage(argv[0]); return 1; }
    }
    if (model_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    laya_context_params mp = laya_context_default_params();

    laya_model * model   = laya_model_load(model_path.c_str());
    if (!model) {
        return 1;
    }

    if (tokenize) {
        std::vector<int32_t> ids(tokenize_text.size() + 16);
        const int32_t n = laya_tokenize(model, tokenize_text.c_str(), ids.data(), (int32_t) ids.size());
        if (n < 0) {
            return 1;
        }
        std::cout << json(std::vector<int32_t>(ids.begin(), ids.begin() + n)).dump() << "\n";
        laya_model_free(model);
        return 0;
    }

    laya_context_params cp = laya_context_default_params();
    cp.n_threads           = threads;
    laya_context * ctx     = sequence ? nullptr : laya_context_new(model, cp);
    if (!sequence && !ctx) {
        return 1;
    }

    std::stringstream ss;
    if (file.empty()) {
        ss << std::cin.rdbuf();
    } else {
        std::ifstream f(file);
        if (!f) {
            fprintf(stderr, "error: cannot open %s\n", file.c_str());
            return 1;
        }
        ss << f.rdbuf();
    }
    std::vector<std::string> inputs;
    if (jsonl) {
        std::string line;
        while (std::getline(ss, line)) {
            if (line.find_first_not_of(" \t\r") != std::string::npos) inputs.push_back(line);
        }
    } else {
        inputs.push_back(ss.str());
    }

    int rc = 0;
    for (const auto & in : inputs) {
        try {
            const json req = json::parse(in);
            if (sequence) {
                const json &      st    = req.contains("state") ? req["state"] : json("");
                const std::string state = st.is_string() ? st.get<std::string>() : py_dumps(st, false);
                json              out   = json::object();
                for (auto it = req["questions"].begin(); it != req["questions"].end(); ++it) {
                    question_storage s;
                    std::string      err;
                    if (!parse_question(it.key(), it.value(), s, err)) throw std::runtime_error(err);
                    std::vector<int32_t> ids(laya_model_max_len(model)), mk(std::max(1, s.q.n_options) + 2);
                    const int32_t n = laya_build_sequence(model, state.c_str(), &s.q, ids.data(), (int32_t) ids.size(), mk.data());
                    if (n < 0) return 1;
                    const int n_mk = s.q.type == LAYA_QTYPE_NOUL ? 2 : s.q.n_options;
                    out[it.key()]  = { { "ids", std::vector<int32_t>(ids.begin(), ids.begin() + n) },
                                       { "markers", std::vector<int32_t>(mk.begin(), mk.begin() + n_mk) } };
                }
                std::cout << out.dump() << std::endl;
                continue;
            }
            json resp;
            const auto t0 = std::chrono::steady_clock::now();
            const int  reps = std::max(1, bench);
            for (int r = 0; r < reps; ++r) {
                resp = handle_request(ctx, model, req, logits);
            }
            if (bench > 0) {
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
                fprintf(stderr, "bench: %d questions, %d tokens, %.1f ms/request (%.0f tok/s)\n",
                        (int) req["questions"].size(), resp["usage"]["input_tokens"].get<int>(), ms,
                        resp["usage"]["input_tokens"].get<int>() / (ms / 1000.0));
            }
            std::cout << (jsonl ? resp.dump() : resp.dump(2)) << std::endl;
        } catch (const std::exception & e) {
            fprintf(stderr, "error: %s\n", e.what());
            if (jsonl) {
                std::cout << json{ { "error", e.what() } }.dump() << std::endl;
            }
            rc = 1;
        }
    }

    laya_context_free(ctx);
    laya_model_free(model);
    return rc;
}
