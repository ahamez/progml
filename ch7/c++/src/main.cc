#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <boost/endian/conversion.hpp>
#include <eigen3/Eigen/Dense>

using matrix_type = Eigen::MatrixXd;
//using matrix_type = Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>;

std::uint32_t
read_uint32(std::istreambuf_iterator<char>& isb)
{
  auto value = std::uint32_t{};

  std::copy_n(isb, sizeof(std::uint32_t), reinterpret_cast<char*>(&value));
  ++isb;

  return boost::endian::big_to_native(value);
}

matrix_type
load_images(const std::filesystem::path& path)
{
  auto file = std::ifstream{path};
  if (not file.is_open())
  {
    throw "Invalid path";
  }

  auto isb = std::istreambuf_iterator<char>{file};

  const auto magic_number = read_uint32(isb);
  if (magic_number != 2051)
  {
    throw "Invalid magic number";
  }

  const auto nb_images = read_uint32(isb);
  const auto image_rows = read_uint32(isb);
  const auto image_cols = read_uint32(isb);

  // +1 for bias column
  auto matrix = matrix_type{nb_images, image_rows * image_cols + 1};

  for (auto image = std::uint32_t{0}; image < nb_images; ++image)
  {
    matrix(image, 0) = 1; // bias
    for (auto i = std::uint32_t{1}; i < image_rows * image_cols + 1; ++i)
    {
      matrix(image, i) = *isb;
      ++isb;
    }
  }

  return matrix;
}

matrix_type
load_labels(const std::filesystem::path& path)
{
  auto file = std::ifstream{path};
  if (not file.is_open())
  {
    throw "Invalid path";
  }

  auto isb = std::istreambuf_iterator<char>{file};

  const auto magic_number = read_uint32(isb);
  if (magic_number != 2049)
  {
    throw "Invalid magic number";
  }

  const auto nb_labels = read_uint32(isb);

  auto matrix = matrix_type{nb_labels, 1};

  for (auto label = std::uint32_t{0}; label < nb_labels; ++label)
  {
    matrix(label, 0) = *isb;
    ++isb;
  }

  return matrix;
}

matrix_type
one_hot_encode(const matrix_type& vector)
{
  Eigen::MatrixXd mat = Eigen::MatrixXd::Constant(10, 10, 1.0);
  mat(0, 0) = 1;

  constexpr auto nb_classes = 10;
  const auto nb_labels = vector.rows();
  matrix_type matrix = matrix_type::Zero(nb_labels, nb_classes);

  for (auto i = 0; i < nb_labels; ++i)
  {
    const auto label = vector(i, 0);
    matrix(i, label) = 1;
  }
  return matrix;
}

void
to_pgm(std::ostream& os, const matrix_type& matrix, std::size_t image, std::size_t image_rows,
       std::size_t image_cols)
{
  os << "P2\n";
  os << image_cols << ' ' << image_rows << '\n';
  os << "255\n";
  for (auto i = size_t{0}; i < image_rows; ++i)
  {
    for (auto j = size_t{0}; j < image_cols; ++j)
    {
      // +1 to skip bias column
      os << 255 - matrix(image, 1 + i*image_cols + j) << " ";
    }
    os << '\n';
  }
}

matrix_type
sigmoid(const matrix_type& m)
{
  return m.unaryExpr([](double z)
  {
    return 1 / (1 + std::exp(-z));

  });
}

matrix_type
forward(const matrix_type& X, const matrix_type& w)
{
  const auto weight_sum = X * w;
  return sigmoid(weight_sum);
}

matrix_type
argmax(const matrix_type& m)
{
  const auto nb_rows = m.rows();

  auto max_indices = matrix_type{nb_rows, 1};
  for (auto i = 0; i < 3; ++i)
  {
    auto index = matrix_type::Index{nb_rows};
    m.row(i).maxCoeff(&index);
    max_indices(i, 0) = index;
  }

  return max_indices;
}

matrix_type
classify(const matrix_type& X, const matrix_type& w)
{
  const auto y_hat = forward(X, w);
  const auto labels = argmax(y_hat);

  return labels;
}

double
loss(const matrix_type& X, const matrix_type& Y, const matrix_type& w)
{
  const auto y_hat = forward(X, w);
  const auto first_term = Y * y_hat.unaryExpr([](double z){return std::log(z);});

  const auto one_minus_Y = Y.unaryExpr([](double z){return 1 - z;});
  const auto log_one_minus_y_hat = y_hat.unaryExpr([](double z){return std::log(1 - z);});
  const auto second_term = one_minus_Y * log_one_minus_y_hat;

  return -(first_term + second_term).sum() / X.rows();
}

matrix_type
gradient(const matrix_type& X, const matrix_type& Y, const matrix_type& w)
{
  const auto nb_examples = X.rows();

  const auto X_t = X.transpose();
  const auto prediction = forward(X, w);
  const auto diff = prediction - Y;
  const auto mult = X_t * diff;

  return mult.unaryExpr([=](double z){return z / nb_examples;});
}

void
report(std::size_t iteration, const matrix_type& X_train, const matrix_type& Y_train, const matrix_type& X_test, const matrix_type& Y_test, const matrix_type& w)
{
//  const auto classified = classify(X_test, w);
  const auto loss_percent = loss(X_train, Y_train, w);
  std::cout << iteration << " loss: " << loss_percent << "%\n";
}

matrix_type
train(const matrix_type& X_train, const matrix_type& Y_train, const matrix_type& X_test, const matrix_type& Y_test, std::uint32_t iterations, double lr)
{
  matrix_type w = matrix_type::Zero(X_train.cols(), Y_train.cols());
  for (auto i = std::uint32_t{0}; i < iterations; ++i)
  {
    report(i, X_train, Y_train, X_test, Y_test, w);
    const auto g = gradient(X_train, Y_train, w).unaryExpr([=](auto x){return x * lr;});
    w -= g;
  }
  report(iterations, X_train, Y_train, X_test, Y_test, w);
  return w;
}

int
main(int argc, const char** argv)
{
  if (argc < 2)
  {
    std::cerr << "Path to dataset missing\n";
    return -1;
  }

  const auto dataset_path = std::filesystem::path{argv[1]};

  if (not exists(dataset_path))
  {
    std::cerr << "Invalid dataset directory\n";
    return -1;
  }

  const auto start_time = std::chrono::steady_clock::now();

  const auto x_train = load_images(dataset_path/"train-images-idx3-ubyte");
  const auto x_test = load_images(dataset_path/"t10k-images-idx3-ubyte");

  const auto y_train_unencoded = load_labels(dataset_path/"train-labels-idx1-ubyte");
  const auto y_train = one_hot_encode(y_train_unencoded);
  const auto y_test = load_labels(dataset_path/"t10k-labels-idx1-ubyte");

  const auto end_time = std::chrono::steady_clock::now();
  const auto loading_time = std::chrono::duration<double>{end_time - start_time};

  //  auto pgm_file = std::ofstream{"/Users/hal/Desktop/foo.pgm"};
  //  to_pgm(pgm_file, x_train, 1'000, 28, 28);

  std::cout << "Loading time: " << loading_time.count() << '\n';

  const auto w = train(x_train, y_train, x_test, y_test, 1'000, 1.0e-5);


  return 0;
}
