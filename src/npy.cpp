#include "npy.hpp"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cstring>

std::vector<size_t> NpyArray::strides() const {
    std::vector<size_t> s(shape.size());
    size_t acc = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        s[i] = acc;
        acc *= shape[i];
    }
    return s;
}

void NpyArray::squeeze() {
    std::vector<size_t> new_shape;
    for (size_t d : shape) {
        if (d != 1) new_shape.push_back(d);
    }
    // numpy.squeeze() on an all-ones-shape array yields a 0-d array;
    // we keep at least one dimension to stay usable as a matrix/vector.
    if (new_shape.empty()) new_shape.push_back(1);
    shape = new_shape;
}

namespace {

struct DType {
    char kind;   // 'f' float, 'i' signed int, 'u' unsigned int, 'b' bool
    int itemsize;
};

DType parse_descr(const std::string& descr) {
    // descr looks like "<f8", "<f4", "<i4", "|u1", "=f8", etc.
    if (descr.empty()) throw std::runtime_error("npy: empty descr");
    char byteorder = descr[0];
    if (byteorder == '>') {
        throw std::runtime_error("npy: big-endian arrays are not supported");
    }
    size_t pos = 0;
    if (byteorder == '<' || byteorder == '=' || byteorder == '|') pos = 1;
    if (pos >= descr.size()) throw std::runtime_error("npy: malformed descr '" + descr + "'");
    char kind = descr[pos];
    int itemsize = std::stoi(descr.substr(pos + 1));
    return DType{kind, itemsize};
}

std::string find_field(const std::string& header, const std::string& key) {
    size_t k = header.find("'" + key + "'");
    if (k == std::string::npos) throw std::runtime_error("npy: header missing '" + key + "'");
    size_t colon = header.find(':', k);
    size_t valstart = colon + 1;
    while (valstart < header.size() && (header[valstart] == ' ')) valstart++;
    // Value ends at the next top-level comma. Track paren depth so tuples
    // like "(100, 8)" aren't split early.
    int depth = 0;
    size_t i = valstart;
    for (; i < header.size(); i++) {
        char c = header[i];
        if (c == '(' || c == '[') depth++;
        else if (c == ')' || c == ']') {
            if (depth == 0) break;
            depth--;
        } else if (c == ',' && depth == 0) {
            break;
        }
    }
    std::string val = header.substr(valstart, i - valstart);
    return val;
}

std::vector<size_t> parse_shape(const std::string& tuple_str) {
    std::vector<size_t> shape;
    std::string s = tuple_str;
    // strip parens
    size_t open = s.find('(');
    size_t close = s.rfind(')');
    if (open == std::string::npos || close == std::string::npos) {
        throw std::runtime_error("npy: malformed shape tuple '" + tuple_str + "'");
    }
    s = s.substr(open + 1, close - open - 1);
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            if (!cur.empty()) {
                shape.push_back((size_t)std::stoull(cur));
                cur.clear();
            }
        } else if (!std::isspace((unsigned char)c)) {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) shape.push_back((size_t)std::stoull(cur));
    return shape;
}

template <typename T>
void read_typed(std::ifstream& f, std::vector<double>& out, size_t n) {
    std::vector<T> buf(n);
    f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)(n * sizeof(T)));
    if (!f) throw std::runtime_error("npy: unexpected EOF while reading data");
    out.resize(n);
    for (size_t i = 0; i < n; i++) out[i] = (double)buf[i];
}

} // namespace

NpyArray load_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy: cannot open file '" + path + "'");

    char magic[6];
    f.read(magic, 6);
    if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("npy: bad magic in '" + path + "'");
    }
    uint8_t major = 0, minor = 0;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t hl16 = 0;
        f.read(reinterpret_cast<char*>(&hl16), 2);
        header_len = hl16;
    } else {
        f.read(reinterpret_cast<char*>(&header_len), 4);
    }
    if (!f) throw std::runtime_error("npy: truncated header in '" + path + "'");

    std::string header(header_len, '\0');
    f.read(&header[0], header_len);
    if (!f) throw std::runtime_error("npy: truncated header in '" + path + "'");

    std::string descr = find_field(header, "descr");
    // descr comes back quoted, e.g. "'<f8'"
    {
        size_t q1 = descr.find('\'');
        size_t q2 = descr.rfind('\'');
        if (q1 == std::string::npos || q2 == q1) throw std::runtime_error("npy: malformed descr field");
        descr = descr.substr(q1 + 1, q2 - q1 - 1);
    }
    std::string fortran_str = find_field(header, "fortran_order");
    bool fortran_order = fortran_str.find("True") != std::string::npos;
    std::string shape_str = find_field(header, "shape");

    NpyArray arr;
    arr.shape = parse_shape(shape_str);
    size_t n = 1;
    for (size_t d : arr.shape) n *= d;
    if (arr.shape.empty()) n = 1; // 0-d array: single scalar

    DType dt = parse_descr(descr);
    std::vector<double> flat;
    if (dt.kind == 'f' && dt.itemsize == 8) read_typed<double>(f, flat, n);
    else if (dt.kind == 'f' && dt.itemsize == 4) read_typed<float>(f, flat, n);
    else if (dt.kind == 'i' && dt.itemsize == 8) read_typed<int64_t>(f, flat, n);
    else if (dt.kind == 'i' && dt.itemsize == 4) read_typed<int32_t>(f, flat, n);
    else if (dt.kind == 'i' && dt.itemsize == 2) read_typed<int16_t>(f, flat, n);
    else if (dt.kind == 'i' && dt.itemsize == 1) read_typed<int8_t>(f, flat, n);
    else if (dt.kind == 'u' && dt.itemsize == 8) read_typed<uint64_t>(f, flat, n);
    else if (dt.kind == 'u' && dt.itemsize == 4) read_typed<uint32_t>(f, flat, n);
    else if (dt.kind == 'u' && dt.itemsize == 2) read_typed<uint16_t>(f, flat, n);
    else if (dt.kind == 'u' && dt.itemsize == 1) read_typed<uint8_t>(f, flat, n);
    else if (dt.kind == 'b') read_typed<uint8_t>(f, flat, n);
    else throw std::runtime_error("npy: unsupported dtype '" + descr + "'");

    if (!fortran_order || arr.shape.size() <= 1) {
        arr.data = std::move(flat);
    } else {
        // Re-order from column-major (Fortran) to row-major (C) storage.
        arr.data.resize(n);
        std::vector<size_t> c_strides = arr.strides(); // row-major strides for target
        std::vector<size_t> f_strides(arr.shape.size());
        size_t acc = 1;
        for (size_t i = 0; i < arr.shape.size(); i++) {
            f_strides[i] = acc;
            acc *= arr.shape[i];
        }
        std::vector<size_t> idx(arr.shape.size(), 0);
        for (size_t lin = 0; lin < n; lin++) {
            // decompose lin as a Fortran-order linear index into idx[]
            size_t rem = lin;
            for (size_t i = 0; i < arr.shape.size(); i++) {
                idx[i] = (arr.shape[i] == 0) ? 0 : (rem / f_strides[i]) % arr.shape[i];
            }
            size_t c_lin = 0;
            for (size_t i = 0; i < arr.shape.size(); i++) c_lin += idx[i] * c_strides[i];
            arr.data[c_lin] = flat[lin];
        }
    }
    return arr;
}
